# 架构

## 分层

```
┌─ cli.py ─────────────────────────────────────────────────────────────┐
│ demo | attr | judge | run | validate | list-rules | weights-check     │
└───────┬──────────────────────┬───────────────────┬───────────────────┘
        │                      │                   │
┌───────▼────────┐   ┌─────────▼─────────┐  ┌──────▼──────────────┐
│ odd_suite/     │   │ odd_suite/        │  │ judge/odd_judge/     │
│ adapters/      │   │ pipeline/         │  │ (submodule engine)   │
│ ImageAttr      │   │ config.py         │  │ OddJudge             │
│ NuScenesDet    │──►│ runner.py         │─►│ evaluate(obs)         │
│ JsonSidecar    │   └─────────┬─────────┘  │ → JudgmentReport      │
│ Mock/Demo      │             │            └───────────────────────┘
└───────┬────────┘   ┌─────────▼─────────┐
        │            │ mapping/          │
        │            │ env_mapper        │  纯 Python，无 torch
        │            │ dynamic_rollup    │  映射表 JSON 可覆写
        │            │ sensor_defaults   │
        │            │ builder           │
        │            │ registry          │
        │            └─────────┬─────────┘
        │                      │
┌───────▼────────┐   ┌─────────▼─────────┐
│ vendor/attr/   │   │ Observation       │
│ (torch 惰性)    │   │ fields.{scenery,  │
│ vendor/sparse4d│   │ vehicle,          │
│ (子进程调用)     │   │ environment,      │
└────────────────┘   │ dynamic,sensors}  │
                     └───────────────────┘
```

## 依赖边界（重要）

- **核心层零第三方依赖**：`schema`、`mapping`、`pipeline` 胶水、`odd_judge`、
  CLI 的 `demo` / `judge` / `validate` / `list-rules` / `weights-check`、
  `run --no-attr --detections-json`。实测 `import odd_suite` 不触发 torch。
- `torch` 只在 `adapters/attr_adapter.py`、`vendor/attr/checkpoint.py` 被**惰性导入**；
  缺失时抛 `MissingDependencyError`（提示 `pip install 'odd-suite[attr]'`）。
- `vendor/sparse4d` 只在 `Sparse4DDetector` 内以**子进程**方式调用
  `run.py infer`，不污染当前进程的 Hydra/torch 环境。

## 数据流

1. **属性分支**：图像 → `MultiTaskYOLO`（5 头）→ `AttrResult`
   → `EnvMapper`（映射表）→ `environment.*`。
   多标签 weather 逐项独立触发；Fog 把 `visibility_m` 钳到 ≤350，
   Haze/Dust 钳到 ≤200（消解矛盾读数，文档化 tie-breaker）。
2. **检测分支**：检测框（实时 Sparse4D 子进程或离线 nuScenes JSON）
   → `DynamicRollup`（类别→角色）→ `dynamic.*`；`SensorDefaults` → `sensors.*`。
   车流（car/bus/truck/trailer/construction_vehicle）**不计入** obstacles，
   避免 TJP 域内正常车流误触 `obstacles_over_0_3m<=0` 的越界规则。
3. **组装**：`ObservationBuilder` 合并各字段组，优先级 sidecar > 显式 > 映射默认。
4. **判决**：`OddJudge.evaluate(observation)` → `JudgmentReport`
   （状态 × 严重度 × 建议 + 逐规则裕度/告警码，码值见 [docs/warning_codes.md](warning_codes.md)）。
5. **报告**：`report/render.py` 输出合订文本/JSON。

## 不可变性

`AttrResult` / `Detection3D` / `PipelineConfig` / `PipelineResult` 均为
frozen dataclass；映射表在构造时加载，运行期只读。
