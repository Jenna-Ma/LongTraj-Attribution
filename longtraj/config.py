"""Config-file loading for LLM credentials.

Keeps keys out of the shell history and out of the repo. Resolution order for
every field: explicit argument > config file > environment variable > default.

    LongTraj/config.json          <- default location, git-ignored

    {
      "base_url":   "https://api.openai.com/v1",
      "api_key":    "sk-...",
      "model_name": "gpt-4o",

      "temperature": 0.0,          // optional
      "max_tokens":  4096,         // optional
      "timeout":     240,          // optional
      "retries":     3             // optional
    }

Field aliases are accepted so a config copied from a sibling project still
works: ``model`` / ``model_name``, ``base_url`` / ``api_base`` / ``openai_api_base``,
``api_key`` / ``openai_api_key``.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_ENV = "LONGTRAJ_CONFIG"
DEFAULT_NAMES = ("config.json", "longtraj.config.json")

_ALIASES = {
    "model": ("model_name", "model", "llm_model", "detector_model"),
    "base_url": ("base_url", "api_base", "openai_api_base", "openai_base_url",
                 "endpoint"),
    "api_key": ("api_key", "openai_api_key", "key", "token"),
    "temperature": ("temperature",),
    "max_tokens": ("max_tokens", "max_output_tokens"),
    "timeout": ("timeout", "timeout_seconds"),
    "retries": ("retries", "max_retries"),
    # Whether the endpoint's strict JSON mode (response_format json_object) is
    # usable with this model. Default on. Turn it OFF for a model that degrades
    # under it: qwen3-235b-a22b on api.vveai.com answers the literal string
    # "spans" in JSON mode and perfect JSON without it (measured 2026-10-10).
    "json_mode": ("json_mode", "strict_json", "response_format_json"),
}


def default_path() -> Optional[str]:
    """`$LONGTRAJ_CONFIG`, else config.json next to the package or in cwd."""
    env = os.getenv(CONFIG_ENV)
    if env:
        return env
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for base in (os.getcwd(), here):
        for name in DEFAULT_NAMES:
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p
    return None


def load_config(path: Optional[str] = None,
                required: bool = False) -> Dict[str, Any]:
    """Read a config file into normalised keys. Missing file -> {}."""
    p = path or default_path()
    if not p:
        if required:
            raise FileNotFoundError(
                "no config file found; create LongTraj/config.json with "
                "base_url / api_key / model_name, or set $%s" % CONFIG_ENV)
        return {}
    if not os.path.exists(p):
        raise FileNotFoundError("config file not found: %s" % p)

    with open(p, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object: %s" % p)
    # tolerate a nested {"llm": {...}} wrapper
    if "llm" in raw and isinstance(raw["llm"], dict):
        merged = dict(raw)
        merged.update(raw["llm"])
        raw = merged

    lowered = {str(k).lower(): v for k, v in raw.items()}
    out: Dict[str, Any] = {"_path": os.path.abspath(p)}
    for field, names in _ALIASES.items():
        for n in names:
            if lowered.get(n) not in (None, ""):
                out[field] = lowered[n]
                break

    unknown = [k for k in lowered
               if k not in {n for names in _ALIASES.values() for n in names}
               and not k.startswith("_") and k != "llm"]
    if unknown:
        logger.debug("ignoring unknown config keys: %s", ", ".join(sorted(unknown)))
    return out


def describe(cfg: Dict[str, Any]) -> str:
    """One line for logs — the key is masked."""
    if not cfg:
        return "no config file (using environment)"
    key = str(cfg.get("api_key") or "")
    masked = (key[:6] + "..." + key[-4:]) if len(key) > 12 else ("set" if key else "MISSING")
    return "config %s: model=%s base_url=%s api_key=%s" % (
        cfg.get("_path"), cfg.get("model"), cfg.get("base_url"), masked)
