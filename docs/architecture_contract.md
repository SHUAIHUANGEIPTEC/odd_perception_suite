# 架构契约：感知 → 判决的数据通路

回答四个问题：感知结果是否直通判决器？是否经过协议通信？可拓展性体现在哪？缺少信息时哪些判据开启、哪些关闭？

配套语义文档：[`semantics.md`](semantics.md)（三态判决）、[`mapping.md`](mapping.md)（字段映射）。
本文只描述**现状契约**，不引入新行为（缺失策略 `missing_policy` 是本文唯一新增的机制，见第 4 节）。

## 1. 数据通路：不是裸直通，中间有映射与组装层

```
感知适配器                     映射/组装                     判决
─────────────                 ─────────────                 ─────────────
ImageAttrAdapter  ─┐          EnvMapper                  OddJudge
NuScenesDetAdapter ┼─ 字段组 ─→ DynamicRollup ─ Observation ─→ evaluate(rule)
JsonSidecarAdapter ┤          SensorDefaults                JudgmentReport
MockAdapter       ─┘          ObservationBuilder
```

| 阶段 | 位置 | 说明 |
|---|---|---|
| 适配器产出字段组 | `odd_suite/adapters/attr_adapter.py:53`（`environment`）、`detect_adapter.py:67-73`（`dynamic`+`sensors`）、`sidecar_adapter.py:27-33`（任意组） | 模型差异在这里被吸收 |
| 映射 + 合并 | `odd_suite/mapping/env_mapper.py`、`dynamic_rollup.py`、`sensor_defaults.py`、`builder.py:52-68` | 优先级：侧车显式值 > 映射默认值 > 缺失；并派生 `vehicle.overspeed_kmh`（`builder.py:23-44`） |
| 统一观测 | `odd_judge/core/types.py:153` `Observation` | `fields` 嵌套 dict + `get()` 点路径；缺失返回单例哨兵 `VALUE_MISSING`（`types.py:36`） |
| 判决 | `odd_judge/api.py:91` `OddJudge.evaluate` → `core/evaluate.py` 逐规则 | 只认 `Observation` 契约，不认模型输出 |

**结论**：感知结果不直通判决器，但也没有任何通信层；是**进程内直调 + 统一中间表示**。

## 2. 通信现状：有数据契约，没有传输协议

非 `vendor/` 代码中无 socket / HTTP / protobuf / ROS / gRPC / 消息队列的任何引用。

- **唯一的跨进程边界**：Sparse4D 推理走 `subprocess.run`（`odd_suite/adapters/detect_adapter.py:143`），交换 nuScenes submission JSON（`*val_results.json`，`detect_adapter.py:144-147`）。
- **序列化契约**（内存 / 文件，非传输）：

| 契约 | 位置 | 用途 |
|---|---|---|
| `Observation.from_dict` | `odd_judge/core/types.py:170` | JSON / JSONL → 观测 |
| `RuleResult.to_dict` | `odd_judge/core/types.py:135` | 单规则结果 |
| `JudgmentReport.to_dict / to_json` | `odd_judge/core/types.py:196-215` | 判决报告 |
| `pipeline_to_dict / to_json` | `odd_suite/report/render.py:12,49` | 整条流水线结果 |
| `JsonFileAdapter` | `odd_judge/perception/json_stream.py:17-45` | JSON / JSONL 流式读入 |

**何时需要协议层**：只有当判决器要跨进程、跨机、或与实时中间件（ROS2/DDS/自研总线）集成时才需要。届时只需新增一个 `ObservationAdapter`（入）和一个报告序列化出口，引擎与 profile 不用动。当前单机回放、离线评测、CI 场景下引入中间件是纯负担。

## 3. 可拓展性：七个扩展点

| 扩展点 | 位置 | 能加什么 |
|---|---|---|
| `ObservationAdapter` ABC | `odd_judge/perception/base.py:11` | 新感知源（CAN / 地图 / SPaT） |
| `AttrModel` / `Detector3D` / `DetectionParser` 协议 | `odd_suite/adapters/base.py:14-26` | 新属性模型、新 3D 检测器、新预测文件格式 |
| `MapperRegistry.register_env_rule / register_role` | `odd_suite/mapping/registry.py:21-25` | 映射规则、检测类角色 |
| `register_rule_type / register_element` | `odd_judge/config/registry.py:15-35` | 新规则类型、运行时注入元素 |
| profile `--extends` / `--override` | `odd_judge/config/loader.py:141-165` | 追加 / 按 `element_id` 替换元素 |
| 映射表 `--mapping-dir` | `odd_suite/mapping/env_mapper.py:31` | 整目录 JSON 覆写阈值 |
| switches `by_element` / `by_tag` / `by_rule_type` | `odd_judge/api.py:33` | 按维度 / 标签 / 类型开闭判据 |

`applies_when`（`core/evaluate.py:335`）提供条件适用性：条件为假的规则判 `SKIPPED` / `CONDITIONAL-SKIP`，不参与聚合。

## 4. 缺少信息时：不是「开闭判据」，而是保守赋值

先说结论：**不存在按判据自动开关的机制**。观测缺失时每条规则仍会被求值，只是得到一个保守结论；真正的「关闭」只有两处——switches 人工关闭（记 `skipped[]`）与 `applies_when` 不适用（`SKIPPED`）。

### 4.1 处理优先级（高 → 低）

| 优先级 | 机制 | 缺信息时的行为 | 位置 |
|---|---|---|---|
| 1 | `requirement = undefined` | UNDEFINED / `UNDEFINED-REQ`，不参与聚合 | `evaluate.py:76` |
| 2 | 元素级 `undefined_as` | `allowed`→INSIDE ｜ `not_allowed`→OUTSIDE | `evaluate.py:90` |
| 3 | `missing_policy.by_element[id]` | 按取值 | `evaluate.py:117` |
| 4 | `missing_policy.by_tag[tag]`（按 `rule.tags` 顺序首个命中） | 按取值 | `core/types.py` `MissingPolicy.resolve` |
| 5 | `missing_policy.default`（缺省 `"mode"`） | 按取值 / 回落 | 同上 |
| 6 | profile `definition_mode` | `restrictive`→OUTSIDE(`UNDEFINED-STRICT`) ｜ `permissive`→INSIDE(`UNDEFINED-PERMISSIVE`) ｜ `default`→UNDEFINED(`UNDEFINED-DEFAULT`) | `evaluate.py:113-151` |

### 4.2 `missing_policy` 写法

profile 顶层可选字段，取值 ∈ `allowed | not_allowed | undefined | mode`：

```json
"missing_policy": {
  "by_element": {"ODC.2.1.5.speed_limit": "allowed"},
  "by_tag": {"scenery": "undefined", "sensor": "not_allowed"},
  "default": "mode"
}
```

- `undefined` = 判 UNDEFINED，**不参与聚合**（最接近"关闭"的语义）
- `allowed` = 缺信息当作域内（谨慎使用）
- `not_allowed` = 缺信息当作域外（等价 restrictive）
- `mode` = 回落 profile 的 `definition_mode`

命中来源写进 `RuleResult.policy_source` 与 `report.undefined_handled[].policy`，可审计。合并语义（`--extends` / `--override`）：`by_element` / `by_tag` 按 sub-key 覆写，**后者优先**；`default` 后者覆盖前者。

**已知局限**：`by_tag` 是粗粒度近似——`scenery` 标签下既有地图类（`road_type`）也有事件类（`construction_zone`）。要精确到数据来源，用元素级 `undefined_as`（如 `profiles/odc_ext_cn.json` 的限速：缺侧车时 `undefined_as: allowed`，否则 restrictive 会把每帧判成域外）。

### 4.3 完整缺失场景表

| 场景 | 行为 | reason_code |
|---|---|---|
| 元素 `requirement=undefined` | UNDEFINED，不入聚合 | `UNDEFINED-REQ` |
| 元素级 `undefined_as=allowed` | INSIDE | — |
| 元素级 `undefined_as=not_allowed` | OUTSIDE | `UNDEFINED-AS-NOT-ALLOWED` |
| `missing_policy` 命中 `allowed` | INSIDE | `MISSING-POLICY-ALLOWED` |
| `missing_policy` 命中 `not_allowed` | OUTSIDE | `MISSING-POLICY-NOT-ALLOWED` |
| `missing_policy` 命中 `undefined` | UNDEFINED，不入聚合 | `MISSING-POLICY-UNDEFINED` |
| 无 policy，`definition_mode=restrictive` | OUTSIDE（判据仍然生效） | `UNDEFINED-STRICT` |
| 无 policy，`definition_mode=permissive` | INSIDE | `UNDEFINED-PERMISSIVE` |
| 无 policy，`definition_mode=default` | UNDEFINED，不入聚合 | `UNDEFINED-DEFAULT` |
| `applies_when` 为假 | SKIPPED，不入聚合 | `CONDITIONAL-SKIP` |
| switches 关闭（`by_element` / `by_tag` / `by_rule_type` / `global_enabled`） | 记入 `report.skipped[]`，不求值 | — |
| 观测值无法转换（如数值规则拿到非数字） | 同「观测缺失」路径 | 同上 |

### 4.4 switches 只做人工开闭

`switches/*.json` 消费的键只有五个：`global_enabled`、`by_element`、`by_tag`、`by_rule_type`、`boundary_warnings_enabled`（`api.py:33-37`）。
历史遗留的 `on_switch_mismatch` 与 `boundary_severity` 是**死配置**（无任何代码读取），已删除；`tests/test_missing_policy.py::test_switches_config_has_no_dead_keys` 固定住这个结论，新增键必须同步被 `SwitchPolicy` 消费。
