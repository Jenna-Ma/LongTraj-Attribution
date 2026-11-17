"""OpenAI-compatible LLM client with disk cache, retries and JSON coercion.

Credentials come from a config file (see ``longtraj.config``), falling back to
the environment:

    LongTraj/config.json    {"base_url": ..., "api_key": ..., "model_name": ...}
    OPENAI_API_KEY / OPENAI_BASE_URL / LONGTRAJ_MODEL
    LONGTRAJ_API_KEY / LONGTRAJ_BASE_URL   (override the OPENAI_* pair)

Explicit constructor arguments win over both.

The cache is keyed on (model, prompt, temperature, max_tokens) so reruns of
the pipeline over the same corpus cost nothing, which matters because
tracing issues one call per anchor and compression one call per span.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

from .config import load_config

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_DANGLING_THINK = re.compile(r"<think\b[^>]*>.*?$", re.IGNORECASE | re.DOTALL)
_ORPHAN_CLOSE = re.compile(r"^.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)


# Retrying a truncated answer doubles the budget; this bounds that growth so a
# genuinely runaway generation cannot escalate without limit.
MAX_TOKEN_CEILING = 16384

#: Hard ceiling on the prompt, in characters.
#:
#: The 240000 that stood here until 2026-10-13 was measured against
#: ``qwen3-235b-a22b-thinking-2507``, which rejected anything over 126976 INPUT
#: tokens with ``InternalError.Algo.InvalidParameter: Range of input length
#: should be [1, 126976]`` and no retry could clear it. **That model was
#: dropped by api.vveai.com on 2026-10-10** and the figure was never re-measured
#: against its replacement.
#:
#: Re-measured 2026-10-13 against ``qwen3.7-plus`` (probe/cap_probe.py and
#: probe/cap_probe2.py), sending the longest REAL rendered body in the TA
#: corpus (635312 chars) and multiples of it: 120k, 240k, 320k, 420k, 560k,
#: 640k, 800k, 1.0M, 1.4M and 2.0M characters were all accepted with
#: ``finish_reason == "stop"``. The 800k and larger probes carried a passphrase
#: in the LAST line and the model returned it verbatim every time, so the
#: prompt was ingested to the end rather than silently truncated upstream.
#:
#: 1500000 is therefore not a measured refusal point -- nothing refused -- it is
#: a deliberate stop short of the largest thing tried, so a runaway prompt still
#: hits a ceiling here instead of turning into an unbounded bill.
MAX_PROMPT_CHARS = 1500000

#: Errors that no retry and no later call can clear, because the account
#: itself is refusing service: an exhausted balance, a revoked or wrong key.
#: They are worth separating from every other 4xx for one reason -- measured
#: on 2026-09-24, when the balance ran out mid-chain at 03:41, the run kept
#: going for another forty minutes and produced two complete-looking result
#: files built from 1270 failed calls and zero successful ones. Retrying each
#: of those three times bought nothing; the only thing that would have helped
#: is stopping.
#:
#: ``model_not_found`` is in here for the same reason, measured the same way.
#: On 2026-10-10 api.vveai.com dropped qwen3-235b-a22b in the middle of a run.
#: The run ground through 8679 404s and 2168 give-ups over three hours and then
#: wrote a complete-looking sp_hyfx2.jsonl reporting acc@0 = 0.326, built almost
#: entirely from deterministic fallbacks. A number that looks like a reading and
#: is not one is worse than a crash. Strictly the model, not the account, is
#: what refused -- so `account_refused` on such a run means "void", not
#: "unpaid"; the distinction is in the log line, which names the real cause.
#:
#: This gateway has TWO ways of saying the same thing and both are needed: a
#: 404 ``Model not exist`` for a model it lists but cannot serve, and a 403
#: ``暂无可用渠道`` ("no channel available for this model in your group") for one
#: it does not route at all. Neither clears on retry.
_FATAL_ERROR = re.compile(
    r"Arrearage|overdue|insufficient[_ ]quota|exceeded your current quota"
    r"|Access denied, please make sure your account"
    r"|invalid[_ ]api[_ ]key|Incorrect API key|Unauthorized"
    r"|model[_ ]not[_ ]found|Model not exist|暂无可用渠道",
    re.I)

#: Matches the subset of _FATAL_ERROR that is the MODEL's fault, not the
#: account's, so the log line can name the real cause.
_MODEL_GONE = re.compile(
    r"model[_ ]not[_ ]found|Model not exist|暂无可用渠道", re.I)

#: Set the first time a ``_FATAL_ERROR`` is seen. Process-wide on purpose: the
#: account is a property of the process, not of one client or one thread, and
#: every worker should give up at once rather than discovering it separately.
_ACCOUNT_DEAD = threading.Event()


def account_is_dead() -> bool:
    """Whether a fatal account error has been seen in this process."""
    return _ACCOUNT_DEAD.is_set()


def _clip_middle(s: str, limit: int) -> str:
    """Head and tail of ``s``, dropping the middle. Local to avoid an import.

    ``schema.clip_middle`` does the same thing, but ``llm`` is imported by
    nearly every module and is deliberately kept free of intra-package
    dependencies beyond ``config``.
    """
    if len(s) <= limit:
        return s
    half = max(1, (limit - 40) // 2)
    return "%s\n... [%d chars dropped] ...\n%s" % (
        s[:half], len(s) - 2 * half, s[-half:])

# Temperature used for every sample above variant 0. Self-consistency needs
# the draws to actually differ; at the configured default of 0.0 they would
# be k copies of one answer and the vote would be decoration.
SAMPLE_TEMPERATURE = 0.7


def _split_response(res: Any) -> Any:
    """Accept ``(text, finish, usage)``, ``(text, finish)``, or a bare string.

    Subclasses and test stubs override ``_post`` and predate both the tuple
    and the usage dict, so keep reading them rather than breaking on a
    missing field. The third slot is the endpoint's own token accounting;
    ``None`` means it did not report any, which is NOT the same as zero and
    is counted separately so a partial-coverage average can never be read as
    a complete one.
    """
    if isinstance(res, tuple):
        text, finish, usage = (list(res) + [None, None])[:3]
        return str(text or ""), finish, usage
    return str(res or ""), None, None


def _token_usage(raw: Any) -> Optional[Dict[str, int]]:
    """Normalise one response's usage block to plain ints.

    Gateways disagree on both the container (dict vs object) and the names,
    so read either and fall back to the OpenAI spelling. A block that
    carries no recognisable count returns ``None`` rather than a dict of
    zeros -- a call whose tokens are unknown must not dilute the mean.
    """
    if raw is None:
        return None

    def _get(*names):
        for n in names:
            v = (raw.get(n) if isinstance(raw, dict)
                 else getattr(raw, n, None))
            if isinstance(v, (int, float)):
                return int(v)
        return None

    p = _get("prompt_tokens", "input_tokens")
    c = _get("completion_tokens", "output_tokens")
    t = _get("total_tokens")
    if p is None and c is None and t is None:
        return None
    if t is None:
        t = (p or 0) + (c or 0)
    # ``completion_tokens`` is the BILLED completion, which on a reasoning
    # model includes tokens that never reach the body: qwen3.7-plus answers
    # "OK" with 173 completion tokens, 169 of them reasoning. That is why the
    # count has to come from here and not from len(text)/N -- measured on the
    # two best arms, completion runs 0.35-0.54 body chars per billed token,
    # so any chars-based estimate understates it by 6-10x. Reasoning tokens
    # are left folded into the completion figure rather than split out,
    # because that is the number that gets charged.
    return {"prompt_tokens": p or 0, "completion_tokens": c or 0,
            "total_tokens": t}


def _looks_like_json(text: str) -> bool:
    s = strip_think(str(text or "")).lstrip()
    s = _FENCE_RE.sub("", s).lstrip()
    return s.startswith("{") or s.startswith("[")


def _truncated(text: str, finish: Optional[str], force_json: bool) -> bool:
    """Was this response cut off rather than finished?

    ``finish_reason == "length"`` is the endpoint saying so outright. When the
    field is missing (some gateways drop it), a body that opens like JSON but
    does not parse is the same failure seen from the other side.
    """
    if finish == "length":
        return True
    if force_json and _looks_like_json(text) and extract_json(text) is None:
        return True
    return False


def strip_think(text: str) -> str:
    """Remove reasoning-model <think> blocks, tolerating unbalanced tags."""
    s = str(text or "")
    s = _THINK_RE.sub("", s)
    low = s.lower()
    if "<think" in low and "</think" not in low:
        s = _DANGLING_THINK.sub("", s)
    elif "</think" in low and "<think" not in low:
        s = _ORPHAN_CLOSE.sub("", s)
    return s.strip()


def extract_json(payload: str) -> Optional[Any]:
    """Pull the first balanced JSON object/array out of a model response."""
    s = _FENCE_RE.sub("", strip_think(payload)).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except Exception:
                        break
    return None


class LLM:
    """Thin wrapper over an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, timeout: Optional[int] = None,
                 retries: Optional[int] = None,
                 cache_dir: Optional[str] = None, dry_run: bool = False,
                 config: Optional[str] = None,
                 config_required: bool = False):
        # resolution order: explicit argument > config file > env > default
        cfg = load_config(config, required=config_required)
        self.config_path = cfg.get("_path")

        def pick(arg, field, env_names, default):
            if arg is not None:
                return arg
            if cfg.get(field) not in (None, ""):
                return cfg[field]
            for n in env_names:
                v = os.getenv(n)
                if v:
                    return v
            return default

        self.model = pick(model, "model",
                          ("LONGTRAJ_MODEL", "DETECTOR_MODEL", "OPENAI_MODEL"),
                          "gpt-4o-mini")
        self.base_url = pick(base_url, "base_url",
                             ("LONGTRAJ_BASE_URL", "OPENAI_BASE_URL"),
                             "https://api.openai.com/v1")
        self.api_key = pick(api_key, "api_key",
                            ("LONGTRAJ_API_KEY", "OPENAI_API_KEY"), "EMPTY")
        self.temperature = float(pick(temperature, "temperature", (), 0.0))
        self.max_tokens = int(pick(max_tokens, "max_tokens", (), 4096))
        self.timeout = int(pick(timeout, "timeout", (), 240))
        self.retries = int(pick(retries, "retries", (), 3))
        # Does this model behave under response_format={"type":"json_object"}?
        # Not every one does: qwen3-235b-a22b on api.vveai.com answers the bare
        # string "spans" -- the first KEY of the requested schema, encoded as a
        # JSON string -- to the unit-segmentation prompt in JSON mode, and
        # returns a correct object for the same prompt without it. Strict mode
        # is a request to the gateway, not a guarantee, so it is a config knob.
        # Off costs nothing when the model is already well-behaved; the callers
        # all go through extract_json either way.
        _jm = pick(None, "json_mode", ("LONGTRAJ_JSON_MODE",), True)
        if isinstance(_jm, str):        # env vars arrive as text; bool("0") is True
            _jm = _jm.strip().lower() not in ("0", "false", "no", "off", "")
        self.json_mode = bool(_jm)
        self.dry_run = dry_run
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "cached": 0, "prompt_chars": 0,
                      "completion_chars": 0, "failures": 0, "empty": 0,
                      "truncated": 0, "blocked": 0,
                      # Reported by the endpoint, not derived from chars.
                      # ``token_calls`` is the number of calls that actually
                      # carried a usage block, so a reader can tell a
                      # complete accounting from a partial one; a cache hit
                      # advances none of these, because no tokens were spent.
                      "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "token_calls": 0}
        self._client = None

    # -- cache -------------------------------------------------------------
    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, key + ".json")

    def _cache_key(self, prompt: str, max_tokens: int, system: str,
                   variant: int = 0) -> str:
        h = hashlib.sha256()
        h.update(("%s|%s|%.3f|%d|%s" % (self.model, system, self.temperature,
                                        max_tokens, prompt)).encode("utf-8"))
        # Variant 0 is the configured temperature and its key is the one every
        # stored run was written under, so the suffix is added only above it.
        # That keeps every existing cache entry valid when sampling is switched
        # on: an A/B against a stored run then pays only for the extra samples.
        if variant:
            h.update(("|sample%d@%.3f" % (variant, SAMPLE_TEMPERATURE))
                     .encode("utf-8"))
        return h.hexdigest()[:40]

    # -- transport ---------------------------------------------------------
    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.timeout)
        except ImportError:
            self._client = "requests"
        return self._client

    def _post(self, prompt: str, max_tokens: int, system: str,
              force_json: bool, temperature: Optional[float] = None) -> Any:
        """Return ``(text, finish_reason, usage)``.

        ``finish_reason`` matters: ``"length"`` means the endpoint stopped
        because it ran out of budget, so the body is a *prefix* of an answer.
        It arrives with HTTP 200 and a non-empty body, so nothing else
        distinguishes it from a complete response.

        ``usage`` is the endpoint's own token count, or ``None`` when it did
        not send one. Everything upstream of this method had only character
        counts to work with, which made every token figure in the repo a
        chars/N estimate; this is the measured quantity.
        """
        client = self._ensure_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: Dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": (self.temperature if temperature is None
                            else float(temperature)),
            "max_tokens": max_tokens,
        }
        if force_json and self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if client == "requests":
            import requests
            r = requests.post(
                self.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer %s" % self.api_key,
                         "Content-Type": "application/json"},
                json=kwargs, timeout=self.timeout)
            if r.status_code >= 400:
                # ``raise_for_status`` reports the status line and throws the
                # body away, but the body is the only place this gateway says
                # WHY. A 4xx that repeats identically on all three attempts is
                # not a transient, and without the body it is undiagnosable
                # from the log -- which is exactly the state four of these left
                # the first run in.
                raise requests.HTTPError(
                    "%s for %s :: %s" % (r.status_code, r.url, r.text[:400]),
                    response=r)
            body = r.json()
            choice = (body.get("choices") or [{}])[0]
            return ((choice.get("message") or {}).get("content") or "",
                    choice.get("finish_reason"), body.get("usage"))
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        # ``usage`` sits on the response, not on the choice -- it covers the
        # whole request, so with n>1 it would be counted once, not per choice.
        return (choice.message.content or "",
                getattr(choice, "finish_reason", None),
                getattr(resp, "usage", None))

    # -- public ------------------------------------------------------------
    def _usable_cache(self, path: str, force_json: bool) -> Optional[str]:
        """Return a cached response only if it is a whole answer.

        Two kinds of poison have to be filtered here, both written by earlier
        runs that could not tell a soft failure from an answer:

        * empty bodies — the endpoint returned HTTP 200 with nothing;
        * truncated bodies — the endpoint stopped at the token ceiling, so the
          text is a *prefix* of JSON that will never parse.

        Replaying either one silently reproduces the lost stage on every rerun,
        which is worse than paying for the call again.
        """
        try:
            entry = json.load(open(path, encoding="utf-8"))
        except Exception:
            return None
        hit = entry.get("response")
        if not (hit or "").strip():
            return None
        if entry.get("finish_reason") == "length":
            return None
        # Legacy entries predate finish_reason, so fall back to content: a body
        # that opens like JSON but does not parse was cut off mid-answer.
        if force_json and _looks_like_json(hit) and extract_json(hit) is None:
            return None
        return hit

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 system: str = "", force_json: bool = False,
                 variant: int = 0) -> str:
        """One completion. ``variant`` draws an independent sample.

        Variant 0 is the configured temperature -- deterministic at the default
        0.0 -- and is what every run on disk was produced with. Any other
        variant is drawn at ``SAMPLE_TEMPERATURE`` and cached under its own key,
        so k samples of one prompt are k cache entries rather than k copies of
        the same answer. Callers that want a vote ask for variants 0..k-1: the
        vote then contains the deterministic answer plus k-1 draws, and k=1
        reduces to exactly the single-sample behaviour.
        """
        mt = max_tokens or self.max_tokens
        if len(prompt) > MAX_PROMPT_CHARS:
            # Above this the endpoint answers 400 with
            # ``Range of input length should be [1, 126976]`` and no retry can
            # help: the ladder in the caller retries an identical over-long
            # body three times and then records a failure, losing the stage
            # outright. Every stage that builds a prompt already caps it far
            # below this (the compressor at 40k, the localiser at 120k), so
            # nothing that works today is touched -- this only catches the tail
            # trajectory that would otherwise return nothing at all. A clipped
            # answer beats a lost stage, but it is a degradation, so say so.
            logger.warning("prompt of %d chars exceeds the endpoint's input "
                           "limit; clipping to %d",
                           len(prompt), MAX_PROMPT_CHARS)
            prompt = _clip_middle(prompt, MAX_PROMPT_CHARS)
        temp = self.temperature if not variant else SAMPLE_TEMPERATURE
        key = self._cache_key(prompt, mt, system, variant)
        path = self._cache_path(key)
        if path and os.path.exists(path):
            hit = self._usable_cache(path, force_json)
            if hit is not None:
                with self._lock:
                    self.usage["cached"] += 1
                return hit
        if self.dry_run:
            return ""
        if _ACCOUNT_DEAD.is_set():
            # The cache lookup above still ran, so a re-run after the account
            # dies replays every stored answer and only skips what it would
            # have had to buy. Counted apart from ``failures`` because it is
            # not one: nothing was attempted.
            with self._lock:
                self.usage["blocked"] += 1
            return ""

        last = None
        budget = mt
        for attempt in range(self.retries):
            try:
                # ``temperature`` is passed only when a sample asked for one
                # that is not the configured default. Subclasses and test stubs
                # override ``_post`` with the four-argument signature, and the
                # unsampled path -- which is every stored run -- must keep
                # reaching them exactly as it always did.
                extra = ({} if temp == self.temperature
                         else {"temperature": temp})
                out, finish, raw_usage = _split_response(
                    self._post(prompt, budget, system, force_json, **extra))
                tok = _token_usage(raw_usage)
                with self._lock:
                    self.usage["calls"] += 1
                    self.usage["prompt_chars"] += len(prompt)
                    self.usage["completion_chars"] += len(out or "")
                    # Counted on every attempt, including the ones the retry
                    # ladder below discards: a truncated or empty body was
                    # still paid for, so dropping its tokens here would report
                    # less than the run actually spent.
                    if tok is not None:
                        self.usage["prompt_tokens"] += tok["prompt_tokens"]
                        self.usage["completion_tokens"] += tok["completion_tokens"]
                        self.usage["total_tokens"] += tok["total_tokens"]
                        self.usage["token_calls"] += 1
                # An empty body is a soft failure, not an answer. Endpoints
                # return one under load with HTTP 200, so no exception fires.
                # Retrying costs one call; accepting it silently loses the
                # stage — and caching it would make the loss permanent.
                #
                # Unless the endpoint says it stopped at the ceiling: a
                # reasoning model can spend the entire budget thinking and
                # emit nothing visible, which arrives as empty *and*
                # finish="length". Retrying that at the same budget can only
                # fail again, so it belongs on the truncation ladder below,
                # which doubles the budget instead.
                if not (out or "").strip() and finish != "length":
                    with self._lock:
                        self.usage["empty"] += 1
                    if attempt < self.retries - 1:
                        logger.warning("empty response (attempt %d/%d); retrying",
                                       attempt + 1, self.retries)
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    logger.error("empty response after %d attempts", self.retries)
                    return ""
                # A body cut off at the token ceiling is also a soft failure.
                # It is non-empty and arrives with HTTP 200, so only
                # finish_reason (or unparseable JSON) gives it away — and
                # caching the prefix makes every later run reuse half an answer.
                if _truncated(out, finish, force_json):
                    with self._lock:
                        self.usage["truncated"] += 1
                    if attempt < self.retries - 1:
                        budget = min(budget * 2, MAX_TOKEN_CEILING)
                        logger.warning("truncated response (attempt %d/%d, "
                                       "finish=%s); retrying with max_tokens=%d",
                                       attempt + 1, self.retries, finish, budget)
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    logger.error("still truncated after %d attempts; returning "
                                 "the partial body uncached", self.retries)
                    return out
                if path:
                    try:
                        json.dump({"response": out, "finish_reason": finish},
                                  open(path, "w", encoding="utf-8"))
                    except Exception:
                        pass
                return out
            except Exception as exc:                       # noqa: BLE001
                last = exc
                if _FATAL_ERROR.search(str(exc)):
                    # Retrying a refused account only multiplies the log. Say
                    # it once, loudly, and let every other worker stop too.
                    if not _ACCOUNT_DEAD.is_set():
                        _ACCOUNT_DEAD.set()
                        logger.error(
                            "ENDPOINT REFUSED (%s) -- no further calls will be "
                            "attempted in this process. Results from here on "
                            "are built from cache only and MUST NOT be read as "
                            "a run: %s",
                            ("model unavailable"
                             if _MODEL_GONE.search(str(exc))
                             else "account refused"),
                            str(exc)[:300])
                    break
                sleep = min(2 ** attempt, 8)
                logger.warning("LLM call failed (attempt %d/%d): %s",
                               attempt + 1, self.retries, exc)
                time.sleep(sleep)
        with self._lock:
            self.usage["failures"] += 1
        logger.error("LLM call gave up after %d attempts: %s", self.retries, last)
        return ""

    def generate_json(self, prompt: str, max_tokens: Optional[int] = None,
                      system: str = "", default: Any = None,
                      force_json: bool = True, variant: int = 0) -> Any:
        raw = self.generate(prompt, max_tokens=max_tokens, system=system,
                            force_json=force_json, variant=variant)
        if not raw:
            return default
        obj = extract_json(raw)
        # A bare scalar is a FAILED parse, not an answer. Every caller of this
        # method asks for an object or a list; none wants a string or a number.
        # The distinction matters because a gateway's strict JSON mode can
        # answer with a well-formed JSON *string* -- qwen3-235b-a22b returns
        # the literal "spans" to the segmentation prompt -- which json.loads
        # accepts, so `obj is None` never fires and the scalar propagates until
        # some caller does .get on it. spans.py:253 crashed on exactly that.
        if obj is None or not isinstance(obj, (dict, list)):
            # one retry without strict JSON mode: some backends degrade under it
            raw2 = self.generate(prompt + "\n\nReturn ONLY valid JSON.",
                                 max_tokens=max_tokens, system=system,
                                 force_json=False, variant=variant)
            obj = extract_json(raw2)
        return obj if isinstance(obj, (dict, list)) else default

    def generate_obj(self, prompt: str, max_tokens: Optional[int] = None,
                     system: str = "", force_json: bool = True,
                     variant: int = 0) -> Dict[str, Any]:
        """``generate_json`` for callers that require a JSON object.

        Always returns a dict, so callers can use ``.get`` unconditionally.

        ``variant`` is forwarded only when it is non-zero. Stubs and subclasses
        in the wild override ``generate_json`` with the old signature, and the
        default path must keep working against them: at variant 0 the call is
        the one they have always received.
        """
        extra = {"variant": variant} if variant else {}
        return as_object(self.generate_json(
            prompt, max_tokens=max_tokens, system=system, default={},
            force_json=force_json, **extra))

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self.usage)


def as_object(obj: Any) -> Dict[str, Any]:
    """Coerce a parsed JSON value to a dict.

    Models occasionally wrap the requested object in a one-element array. Unwrap
    that instead of discarding it: the content is there, only the envelope is
    wrong. Anything else becomes ``{}`` — which every caller already treats as
    "the model gave nothing usable" and answers with a deterministic fallback.
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        dicts = [x for x in obj if isinstance(x, dict)]
        if dicts:
            if len(dicts) > 1:
                logger.info("JSON array held %d objects; using the first",
                            len(dicts))
            return dicts[0]
    return {}


def map_concurrent(fn, items: List[Any], workers: int = 8) -> List[Any]:
    """Ordered concurrent map; exceptions become None instead of propagating."""
    if not items:
        return []
    if workers <= 1:
        out = []
        for it in items:
            try:
                out.append(fn(it))
            except Exception:                              # noqa: BLE001
                logger.exception("task failed")
                out.append(None)
        return out
    from concurrent.futures import ThreadPoolExecutor
    results: List[Any] = [None] * len(items)

    def run(pair):
        i, it = pair
        try:
            results[i] = fn(it)
        except Exception:                                  # noqa: BLE001
            logger.exception("task %d failed", i)
            results[i] = None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run, enumerate(items)))
    return results
