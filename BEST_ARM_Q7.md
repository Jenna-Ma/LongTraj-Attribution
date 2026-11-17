# qwen3.7-plus 最好臂及其结果

生成时间：2026-10-20
数据来源：落盘产物 `probe/*.jsonl` 与 `probe/tr_*/`，**全部零调用复算**。
凡本文件的数字都附了可复现命令；带失败调用的读数一律标注为**下界**。

---

## 一、结论速查

| 语料 | 最好臂 | acc@0 | 干净度 | 配置特征 |
|---|---|---|---|---|
| **TrajAudit (TA)** | **`ta_q7long`** | **0.4902** (50/102) | `failures: 0` ✅ | **简单配置** |
| **SWE-Bench-Pro (SP)** | `sp_fk0` | 0.3605 (31/86) | `failures: 2` ⚠️ **下界** | 复杂配置 |
| SP（干净口径） | `sp_capc1` / `sp_sldry` | 0.3488 (30/86) | `failures: 0` ✅ | 复杂配置 |

> **SP 的两个候选相差仅 1 行，在 4 行噪声地板内 ⇒ 实际无法区分。**
> `sp_fk0` 比 `sp_capc1` 只多 1 行净胜（逐行比对：8 行预测不同，fk0 对 2 行、capc1 对 1 行）。
> **想要干净读数就用 `sp_capc1` 的 0.3488；想要最高数字就用 `sp_fk0` 的 0.3605 并标注下界。**

---

## 二、TA 最好臂 `ta_q7long`

### 2.1 成绩

| | all | dev (48) | test (54) |
|---|---|---|---|
| **acc@0** | **0.4902 (50/102)** | 0.4792 (23/48) | 0.5000 (27/54) |

对照 `ta_capc1`（本轮之前 qwen3.7-plus 在 TA 上的最好臂）0.4412（45/102，dev 0.4167、test 0.4630）
⇒ **全集 +5 行，dev +3、test +2，两半均不掉行。**

调用账本：`calls: 10`，`cached: 1317`，`failures: 0`，`truncated: 0`。
（只发 10 次新调用，因为它与 `ta_q7simple` 只差 `max_prompt_chars`，仅超 120k 的行需重跑。）

### 2.2 配置指纹

```
True 旗标: build_outline keep_diagnostics prior_to_localizer save_spans save_steps span_prior
关键数值: max_prompt_chars=700000  verbatim_limit=2500  prior_samples=1
```

**注意它没有开什么：** `candidate_set` / `candidate_strict` / `verify` / `verify_outdeg` /
`unit_spans` / `prior_compress` / `full_keep` **全部为 False**。
**这正是它赢的原因——见四节。**

### 2.3 完整 acc@k（容差口径，k = 预测与真值相距步数，`dist <= k`）

| k | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| | 0.4902 (50) | 0.5588 (57) | 0.6078 (62) | 0.6275 (64) | 0.6765 (69) | 0.6863 (70) |

acc@0–3 取自产物 `summary`；acc@4/5 按同一公式从 `cases[].distance` 补算。

> **前 k 名口径（交出 k 个答案命中其一）对本臂不存在：** 它 `candidate_set: False`、
> 产物 `candidates` 段为 `None`，每行只交 1 个答案 ⇒ 前 k 名 acc@k ≡ acc@0。

### 2.4 span 相关读数

| 指标 | 值 | 含义 |
|---|---|---|
| **段一致率** | **0.5980 (61/102)** | pred 所在 span == gold 所在 span |
| 随机基线 | 0.1610 | 按段长加权的随机命中概率 |
| **提升比** | **3.71×** | 段一致率 / 基线 |
| 步错但段对 | 11/52 = 0.2115 | 步错的行里，仅 21% 是"段对步错"的近失 |
| `prior_recall` | 0.5484 | 真值落在 span prior 提名的任一区间 |
| `prior_top_recall` | 0.4839 | 落在第一名区间 |
| `prior_coverage` | 0.2081 | 提名区间占全轨迹比例 |
| `prior_followed` | 0.8065 | 定位器是否去了被指的地方 |
| `weak_span_recall` | 0.8529 | 真值落在弱压缩段内 |
| `acc@0 \| prior 命中` | **0.6471** | |
| `acc@0 \| prior 未命中` | **0.2143** | 差 **+0.4328** |

### 2.5 真值在候选池中的比例

**本臂没有候选池**（`candidate_set: False`）。下列为在同一批 102 条轨迹上
**按规则离线建池**的假设性读数：

| 建池方案 | 建出池 | 真值在池 | 真值在池/建出的行 | 池内中位 |
|---|---|---|---|---|
| 旧池 `candidates()` | 89/102 = 0.8725 | 50/102 = 0.4902 | 0.5618 | 10 |
| 新池 `candidates_with_plans(cap=15)` | 89/102 = 0.8725 | 63/102 = 0.6176 | 0.7079 | 12 |

> 旧池的 50/102 与本臂 acc@0 的 50/102 **数值相同但毫无关系**，不是同一批行。

---

## 三、SP 最好臂 `sp_fk0`（及干净对照 `sp_capc1`）

### 3.1 成绩

| 臂 | all | dev (48) | test (38) | failures |
|---|---|---|---|---|
| `sp_fk0` | 0.3605 (31/86) | 0.3542 | 0.3684 | **2 ⇒ 下界** |
| `sp_capc1` | 0.3488 (30/86) | 0.3333 | 0.3684 | 0 |
| `sp_sldry` | 0.3488 (30/86) | 0.3333 | 0.3684 | 0（`calls: 0`，全缓存） |

### 3.2 配置指纹

```
sp_fk0    True: candidate_set candidate_strict keep_diagnostics prior_compress
                prior_to_localizer save_spans save_steps span_prior unit_spans
                verify verify_outdeg
          数值: max_prompt_chars=120000  verbatim_limit=0
          calls=429 cached=2271 failures=2

sp_capc1  同上，另加 full_keep；max_prompt_chars=700000
          calls=15 cached=2687 failures=0
```

**两臂差异只有 `full_keep` 与 `max_prompt_chars`，净 1 行。**

### 3.3 完整 acc@k（容差口径）

| k | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `sp_fk0` | 0.3605 (31) | 0.3605 (31) | 0.4186 (36) | 0.4186 (36) | 0.4419 (38) | 0.4419 (38) |

> **阶梯是结构事实，不是 bug：86 行的 distance 全为偶数，奇数 0 行。**
> 机制：SP 轨迹 `assistant` / `tool` 严格交替（已在产物 `steps[].role` 核实），
> 关键步恒在 `assistant` 侧 ⇒ gold 与 pred 同奇偶（各 85/86 为偶数）。
> **所以 SP 上奇数 k 不含信息：acc@1 ≡ acc@0，acc@3 ≡ acc@2，acc@5 ≡ acc@4。**

### 3.4 span 相关读数

| 指标 | `sp_fk0` | `sp_capc1` |
|---|---|---|
| **段一致率** | **0.4651 (40/86)** | 0.4535 (39/86) |
| 随机基线 | 0.0895 | — |
| **提升比** | **5.19×** | — |
| 步错但段对 | 9/55 = 0.1636 | — |
| `prior_recall` | 0.4651 | 0.4651 |
| `prior_top_recall` | 0.2674 | 0.2674 |
| `prior_coverage` | 0.1827 | 0.1827 |
| `prior_followed` | 0.5349 | 0.5349 |
| `weak_span_recall` | 0.6395 | 0.6395 |
| `acc@0 \| prior 命中` | 0.4750 | 0.4750 |
| `acc@0 \| prior 未命中` | 0.2609 | 0.2391 |

---

## 四、机制：为什么 TA 的最好臂是简单配置

本轮（预注册 `probe/prereg_q7_ta.md`，开跑前锁定、跑完未改）串行跑了三条臂：

| 臂 | 配置 | all | dev | test | failures |
|---|---|---|---|---|---|
| A `ta_q7simple` | `--span-prior --verbatim-limit 2500` | 0.4804 (49) | 0.4583 | 0.5000 | 2 ⇒ 下界 |
| **B `ta_q7long`** | A + `--max-prompt-chars 700000` | **0.4902 (50)** | 0.4792 | 0.5000 | **0** |
| C `ta_q7prior` | `--span-prior` 单开 | 0.4706 (48) | 0.5000 | 0.4444 | 1 ⇒ 下界 |
| 对照 `ta_capc1` | 复杂配置 | 0.4412 (45) | 0.4167 | 0.4630 | 1 ⇒ 下界 |

判决（按预注册 §5 写死的判据，未事后修改）：**A 判平、B 判赢（踩线）、C 判平。**

**关键发现：压低 qwen3.7-plus 的是复杂配置本身，不是模型能力。**
此前 TA 上 5 条 qwen3.7-plus 臂**全部是复杂配置**，跨度仅 4 行（0.4020–0.4412）。
把 `unit_spans` / `prior_compress` / `full_keep` 等中间层开关全部关掉、只留 `--span-prior`，
立刻 +3 行（臂 C）；再解除 120k 截断到 +5 行（臂 B）。**全距从 4 行拉到 9 行。**

可记为仓库第三条规律（中间层改善不兑换终点）的**第 16 次确认**——
本轮这些中间层开关在 qwen3.7-plus 上不仅没兑换，而且**净负**。

### 必须一起讲的三条保留

1. **A/B/C 全距仅 2 行，应读成同一个读数（0.48–0.49 量级），不是三个不同成绩。**
   B 对 A 只差 2 行预测、净 +1 行，远在 4 行噪声地板内。判成一平一赢纯属判据线划在 50 行的离散性。
2. **"比 claude-opus-4-8 的 0.5000 低 1 行"不等于"追平"。** 1 行同样在噪声地板内 ⇒
   **两者在 TA 上目前无法区分**，既不是 claude 仍更好，也不是 qwen3.7-plus 已追平。
3. **oracle 0.5882 不是答案。** 5 条旧臂并集 60/102，但分解为"全对 22 / 全错 38 / 摇摆 42（41.2%）"
   ⇒ **方差形状，不是互补形状**，按预注册禁止第 2 条不得作为"能达到"的成绩上报。

---

## 五、未被采用的轴（本轮未重试，附原因）

| 轴 | 状态 |
|---|---|
| **`--candidates`（TA 上）** | **净负**：`ta_union` 行内对照，有池 55 行 0.4182 vs 无池 43 行 0.4419；新建出池的 30 行 11→4；8 行丢失中 6 行是定位器看到清单后自己改错 |
| verify 轮询（整级，TA 上） | 净负：`ta_verify` 0.4510 < 对照 `ta_anch_off` 0.4804 |
| `--verify-outdeg` | 判正，但 `localize.py` 注释明文 **"MUST NOT BE CLAIMED ON TRAJAUDIT"** |
| 给更多原文 / 更好排版 | 5 次判负，封棺 |
| 改提问形式（E8/E10/E12/E14 + reflect） | 封棺 |
| 多专家融合 / 自洽投票 / 判错重锚 / 分组淘汰 | 均判负 |
| 答案空间放宽或收窄 | 两向均判负 |

---

## 六、复现命令

```bash
cd "C:/maxuyan/my_code/long-horizon/LongTraj-code/LongTraj-code"
export NO_PROXY=api.vveai.com,vveai.com     # 本地代理没起时会静默全灭
export PYTHONIOENCODING=utf-8
python tests/test_no_leak.py                # 开跑前必须全绿

# TA 最好臂。基座只读，必须先 cp -r（write_complete 会写回并污染基座）
rm -rf probe/tr_ta_q7long && cp -r probe/tr_ta_prior probe/tr_ta_q7long
python -u -m longtraj.cli run --source trajaudit --span-prior \
  --verbatim-limit 2500 --max-prompt-chars 700000 --model qwen3.7-plus \
  --workers 8 --span-workers 3 --trace-dir probe/tr_ta_q7long --reuse-flow \
  --out probe/ta_q7long.jsonl --verbose
```

零调用复算（全部读落盘产物）：

```bash
python probe/split_eval.py probe/ta_q7long.jsonl probe/ta_capc1.jsonl   # all/dev/test + 下界告警
python -u probe/span_match.py                                            # 段一致率 + 随机基线
python -u probe/ta_acck.py                                               # 真正的前 k 名（仅带池臂）

# 容差 acc@4/5（summary 只给到 acc@3）
python -c "
import json
b=json.load(open('probe/ta_q7long.jsonl',encoding='utf-8'))
cs=b['cases']; n=len(cs)
for t in range(6):
    h=sum(1 for c in cs if c.get('distance') is not None and c['distance']<=t)
    print('acc@%d = %d/%d = %.4f'%(t,h,n,h/float(n)))
"
```

---

## 七、读这些数字时的陷阱

1. **`acc@1/2/3` 有两种互不相容的口径。** 产物 `summary` 里的是**距离容差**
   （`evaluate.py:49`：`hit@t = bool(have and dist <= t)`，`TOLERANCES = (0,1,2,3)`）；
   **真正的前 k 名只能由 `probe/ta_acck.py` 产出**。两者不可混用、不可并排比较。
2. **SP 上奇数 k 不含信息**（distance 恒为偶数），报 SP 容差曲线应只报偶数 k。
3. **带 `failures` 的行是下界不是读数。** `probe/split_eval.py:76-82` 会自动打印告警，报数时须一并给出。
4. **噪声地板约 4 行。** 小于此的差异不得解读为真实差距。
5. **段一致率必须对着随机基线读。** TA 中位 8 段、SP 中位 17 段，裸比例不可跨语料比较：
   按绝对值 TA 更高（0.60 vs 0.47），按提升比 SP 更高（5.19× vs 3.71×），**两个方向相反，必须同时给**。
6. **`acc@0 | prior 命中` 的巨大落差是相关不是因果。** `evaluate.py:74-78` 自己点明了这个混淆——
   没有"从未见过 prior 的对照"，就无法区分"定位器采纳了提示"与"提示恰好同意了定位器本来要去的地方"。
7. **`verify.pool` 是出处字符串（如 `"candidates+flow"`），不是步号列表**，被投票的步须从
   `verify.verdicts` 读取。

---

## 八、天花板在哪

- TA 上 **49/102 行三条新臂全错**。真值不在流图上的有 33/102（`gold_on_flow` 0.6765），
  而不在流上时 acc 仅 0.273 ⇒ **这是建图质量的天花板，换配置碰不到。**
- 放宽到 span 只多拿约 10 个百分点（TA 0.49→0.60，SP 0.36→0.47）⇒
  **步错的行里 80%+ 不是近失，而是定位到了完全另一个阶段。**
- **想突破 0.50 要从建图质量下手，不要再扫配置。**
