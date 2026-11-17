# LongTraj-Attribution

面向长程智能体失败轨迹的关键步骤定位方法。输入一条最终失败的轨迹，输出最早导致最终失败的 critical step。

本目录归档的是共享主 pipeline：

```text
失败轨迹
  -> 构建数据流 / 控制流图
  -> 划分语义区域并生成区域先验
  -> 按因果相关性压缩轨迹、剪枝候选
  -> 多维候选核验与最终定位
  -> critical step
```

LongRCA episode、WebArena、VitaBench 和 ALFWorld 的语料专用 `probe` 方法不在本快照中。快照来源和边界见 [SNAPSHOT.md](SNAPSHOT.md)。

## 1. 安装

支持 Python 3.9 或更高版本：

```powershell
cd LongTraj-Attribution
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Linux/macOS 将激活命令改为：

```bash
source .venv/bin/activate
```

`pip install -e .` 会同时安装 Python 包和 `longtraj-attribution` 命令。也可以继续使用 `python -m longtraj.integration`。

## 2. API 配置

复制模板并填写 OpenAI-compatible API：

```powershell
Copy-Item config.example.json config.json
```

```json
{
  "base_url": "https://your-endpoint.example/v1",
  "api_key": "YOUR_KEY",
  "model_name": "qwen3.7-plus",
  "temperature": 0.0,
  "max_tokens": 4096,
  "timeout": 240,
  "retries": 3
}
```

凭据只从调用方指定的 `config.json` 读取。该文件已被 `.gitignore` 排除，不应提交到版本库。

## 3. 项目集成入口

### 3.1 Python SDK

在服务启动时创建一个 `AttributionEngine`，后续请求复用同一实例：

```python
from longtraj.integration import AttributionEngine

engine = AttributionEngine(
    profile="sp_qwen_clean",
    config_path="config.json",
    cache_dir=".longtraj_cache",
    trace_dir="traces",
    workers=4,
)

result = engine.locate({
    "task_id": "run-001",
    "task_description": "实现功能并通过测试",
    "steps": [
        {"step": "request", "role": "user", "content": "请实现功能"},
        {
            "step": "plan-1",
            "role": "assistant",
            "name": "Planner",
            "content": "只修改被调模块，不更新调用方",
        },
        {
            "step": "test-output",
            "role": "tool",
            "name": "Computer_terminal",
            "content": "FAILED: 调用方仍使用旧接口",
        },
    ],
})

print(result["critical_step_id"], result["reason"])
```

可用接口：

- `engine.locate(request)`：定位一条 JSON-compatible 轨迹。
- `engine.locate_batch(requests, batch_workers=...)`：批量定位并保持输入顺序。
- `engine.locate_trajectory(trajectory)`：直接接收内部 `Trajectory` 类型。
- `engine.usage()`：取得该 engine 创建以来累计的 API/token/时间统计。
- `locate_failure(request, ...)`：单次便捷调用；在线服务建议复用 engine，避免反复创建客户端。

若要查看 flow、span、prior、candidate 和 verify 等内部产物，调用：

```python
result = engine.locate(request, include_diagnostics=True)
```

### 3.2 输入格式

最小输入结构如下：

```json
{
  "task_id": "唯一任务 ID",
  "task_description": "原始任务描述",
  "steps": [
    {
      "step": "调用方自己的 step ID",
      "role": "user | assistant | tool | system",
      "name": "可选的 agent/tool 名称",
      "content": "该步完整内容"
    }
  ]
}
```

约定：

- `task_id` 必填；同一批次内必须唯一。
- `steps` 必须非空，也可使用别名 `messages`。
- `step` 可为字符串或数字，必须在一条轨迹内唯一；若省略，则使用数组位置。
- `agent`、`model`、`ai` 会归一化为 `assistant`；`function`、`observation` 会归一化为 `tool`。
- 名称为 `Computer_terminal` 的消息按工具输出处理。
- `content` 可以是字符串或 JSON-compatible 对象，后者会被序列化为文本。

严禁把评测标签传入定位入口。`ground_truth`、`critical_step`、`failure_id`、`failure_reason`、`annotation`、`gold_*`、`test_patch` 等字段会被递归检查并直接拒绝。自然语言 `content` 中出现同名字符串不会被误拒绝。

### 3.3 输出格式

```json
{
  "task_id": "run-001",
  "critical_step": 1,
  "critical_step_id": "plan-1",
  "confidence": 0.86,
  "reason": "该计划首次引入了未同步调用方的错误决定",
  "error_type": "planning",
  "selected_step": {
    "index": 1,
    "step_id": "plan-1",
    "role": "assistant",
    "name": "Planner",
    "content": "..."
  },
  "profile": "sp_qwen_clean",
  "model": "qwen3.7-plus",
  "usage": {},
  "compression_ratio": 0.31
}
```

`critical_step` 是 pipeline 使用的、从 0 开始的内部数组位置；`critical_step_id` 是调用方输入的原始 step ID。跨系统关联结果时应优先使用 `critical_step_id`。

### 3.4 JSON CLI

单条轨迹：

```powershell
python -m longtraj.integration `
  --input examples/trajectory.example.json `
  --output result.json `
  --profile sp_qwen_clean `
  --config config.json
```

安装后也可以使用：

```powershell
longtraj-attribution -i trajectory.json -o result.json --profile generic --config config.json
```

CLI 接受以下顶层格式：一条轨迹对象、轨迹数组、`{"trajectory": {...}}` 或 `{"trajectories": [...]}`。输入/输出路径使用 `-` 时分别读取 stdin 或写入 stdout。

只检查接缝而不调用 API：

```powershell
python -m longtraj.integration `
  --input examples/trajectory.example.json `
  --output integration_result.json `
  --profile sp_qwen_clean `
  --dry-run --workers 1
```

`--dry-run` 的预测是本地 fallback，只能用于验证输入、输出和流水线接缝，不能作为定位结果或准确率。

## 4. Profile 选择

| profile | 用途 | 关键设置 |
|---|---|---|
| `generic` | 未知语料的保守起点 | 共享主 pipeline 默认配置 |
| `rootse_qwen_best` | TrajAudit / RootSE | 区域先验、轻量原文恢复、宽 prompt |
| `sp_qwen_highest` | TrajDebug / SWE-bench Pro 已记录最高读数 | 先验条件压缩、严格候选池、多维核验、120k prompt |
| `sp_qwen_clean` | SP 的稳定复现版本 | 与上一项相同，但完整保留候选且使用 700k prompt |

这些 profile 只是固化已记录的配置，不代表跨语料都最优。新项目建议先用 `generic` 小规模验证，再根据轨迹结构选择对应 profile。必要时可通过构造函数的 `config_overrides` 覆盖 `Config` 字段，并保留完整实验记录。

## 5. 主方法与代码对应关系

- `longtraj/tracing.py`：构建数据流/控制流图。
- `longtraj/spans.py`、`longtraj/units.py`、`longtraj/span_prior.py`：语义区域与区域先验。
- `longtraj/compress.py`：KEEP / LIGHT / COMPRESSED 分级压缩。
- `longtraj/candidates.py`、`longtraj/shortlist.py`：候选提取和剪枝。
- `longtraj/localize.py`：主定位器、多维候选核验与融合。
- `longtraj/pipeline.py`：编排完整 pipeline。
- `longtraj/integration.py`：本项目对外的稳定 SDK/JSON CLI 边界。
- `longtraj/artifacts.py`：中间产物和 flow 的持久化。

更多方法细节见 [METHOD.md](METHOD.md)、[METHOD_sp_noout.md](METHOD_sp_noout.md) 和 [BEST_ARM_Q7.md](BEST_ARM_Q7.md)。

## 6. 复现实验

数据不重复放入源码快照。建议目录为：

```text
workspace-root/
  LongTraj-Attribution/
  TrajAudit-main/TrajAudit-main/RootSE/<agent>/<model>/*.json
  TrajDebug-main/TrajDebug-main/data/unified/swebenchpro/*.json
```

使用已发布 flow 复现下游阶段：

```powershell
./run_reproduction.ps1 -Dataset all -FlowMode published -Workers 4
./run_reproduction.ps1 -Dataset rootse -FlowMode published
./run_reproduction.ps1 -Dataset sp -SpArm clean -FlowMode published
```

从原始轨迹重新建图并运行完整方法：

```powershell
./run_reproduction.ps1 -Dataset all -FlowMode fresh -Workers 4
```

无 API 的端到端接缝检查：

```powershell
./run_reproduction.ps1 -Dataset rootse -FlowMode fresh -Limit 1 -DryRun
```

参考结果位于 `reference_results/`，参考 flow 位于 `reference_flows/`。每次新运行会写入独立的 `results/<timestamp>/`，不会修改只读参考产物。

## 7. 测试与真值隔离

```powershell
python tests/test_integration.py
python tests/test_no_leak.py
```

主 pipeline 只接收没有标签字段的 `Trajectory`。标签只存在于评测阶段的独立 `GroundTruth` 对象中；集成入口还会在构造 `Trajectory` 前递归拒绝标签侧字段。

第三方 API 即使温度为 0 仍可能因模型版本或服务端实现产生少量漂移。正式报告结果时应同时保存模型名、endpoint、运行日期、失败调用数、usage 和完整结果文件，而不只记录一个准确率。
