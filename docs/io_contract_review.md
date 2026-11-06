# 《感知输入输出数据定义.xlsx》接口评审 — 修订请求与验收基线

> 评审对象：`/home/shuai/WorkBuddy/workbuddy/output/感知输入输出数据定义.xlsx`（2026-11-03）
> 评审方：`odd_perception_suite`（判据系统实现）。字段级映射见
> [`odd_suite/mapping/tables/io_contract_map.json`](../odd_suite/mapping/tables/io_contract_map.json)
> （由 `tests/test_io_contract_map.py` 守卫，与 profiles 同步）。

## 1. 总体结论

方向与口径**正确**：职责边界（感知=证据生产者，判据=裁决者）、GB/T 45312 + ISO 34503
口径、`out/env_summary` 五任务与本套件属性分类器契约一致，`JudgmentReport.to_dict()`
可直接承载 ISMR 证据包的"逐元素观测+判定+时间戳"。

**但不能原样直连**。判据 profile 实际消费 63 个唯一 `observation_key`，对照结果：

| 状态 | 数量 | 占比 | 含义 |
|---|---|---|---|
| covered | 34 | 54% | 输出表已有对应字段（可含单位换算/布尔派生） |
| convention | 18 | 29% | 有近似来源，需双方约定量化方式/派生式/语义 |
| gap | 11 | 17% | 输出表缺字段，见 §2 修订请求 |

## 2. 给感知侧的 3 条修订请求

### 请求 1：输出表补齐判据已消费的 11 个缺口字段

| 缺口 observation_key | 请求新增的输出字段 | 依据 |
|---|---|---|
| `scenery.accident_scene` / `scenery.construction_zone` | **`out/anomaly`**（输入表已有 `event/anomaly`，**输出表整行缺失**） | 小鹏手册禁用施工/事故 |
| `environment.friction_mu` | `out/env_summary.friction_mu`（输入表 `env/road_surface` 已有 μ，输出丢弃） | 结冰/积雪为 ODD 高危边界 |
| `environment.illumination_lux` | `out/env_summary.illumination_lux`（数值或区间） | profile 按 lux 数值设阈值 |
| `sensors.confidence_camera` / `confidence_lidar` | `out/sensor_conf`（per-sensor / per-task 置信度；当前仅 OOD 三分数） | 判据按通道保守化 |
| `scenery.has_roundabout` / `has_toll_station` / `has_railway_crossing` | `out/map_topology.junction_subtype` / `out/poi_flags`（设施布尔组） | 设施语义输入输出均未覆盖 |
| `scenery.traffic_control` | `out/traffic_control(bool+type)`（人工指挥/临时管制） | 输入输出均未覆盖 |
| `vehicle.dssad_ok` | `out/ego_system.dssad_ok` | DSSAD 自检 |
| `environment.road_temp_c` | `out/env_summary.road_temp_c` | example_enterprise override 消费 |
| `scenery.tunnel_length_m` | `out/map_topology.tunnel_length_m` | 输入有隧道类型无长度 |

### 请求 2：档位字段补"代表值 + 区间"

能见度 4 档、雨雪 0~3 档、闭眼仅有枚举——**档位无法支撑判据的边界带（BOUNDARY 需要
`remaining_margin` 数值距离）**。要求：

- 每个档位字段附 `{representative, low, high}`（如 GE500 → rep=2000, low=500,
  high=∞ 的保守约定；本套件已有 `visibility_env.json` 换算表可作模板）；
- `driver_eyes_closed_s` 给时长数值（DMS 侧累计秒数），非仅状态枚举；
- 判据侧规则：**区间跨阈值 → BOUNDARY；区间无法判定 → 未决(UNDEFINED)**。

### 请求 3：明确"判据消费子集"

输出表含点云 N×6、occupancy 栅格、mask、热图、拓扑图——**这些不进 `Observation`**
（判据只消费标量/枚举/计数）。要求在表中显式标注每字段的消费方：

- `判据`（进 Observation，见 io_contract_map 的 63 键）
- `规划`（freespace/occupancy 等）
- `诊断/复核`（heatmap/drift/explanation）

否则实现方会把张量误塞进判决链路。`ignored_outputs` 已给出本套件的初版判定，可直接回填。

## 3. 双方待决策项（判据侧内部 / 联合）

| # | 冲突 | 建议决议 | 状态 |
|---|---|---|---|
| C1 | xlsx 要求未知类别→未决；本套件 `dynamic_roles.json` 为 `default_role=ignore`（未知被静默忽略） | 感知侧保证类型闭集，**或** unknown 映射为 `dynamic.agent_type=unknown` 走缺省回退 | 待决策 |
| C2 | xlsx 要求 `ROBC < 阈值 → fail-safe/ODD exit`；判据无整帧运行时降级开关 | 约定：上游低分时压低 `sensors.confidence_*` 并切 restrictive profile；整帧开关为功能新增，单独立项 | 待决策 |
| C3 | 「低置信/地图失配→未决」需落 profile `missing_policy` | 提案（**改判据语义，需产品确认**后粘贴进 `profiles/byd_tjp.json` 顶层）：`"missing_policy": {"by_tag": {"map": "undefined"}, "default": "mode"}` — 仅地图来源缺失走未决，其余维持 restrictive（安全优先） | 待决策 |
| C4 | 多源 1~100Hz，无判决节拍与观测过期规则 | 约定：判决节拍 10Hz，逐字段带 timestamp，超时（环境 >2s 等）按缺失回退 | 待约定 |

## 4. 验收基线用法

修订后的 xlsx 即为**接口验收基线**，与既有三道闸叠加：

1. **字段契约**：`io_contract_map.json`（63 键 × 来源/派生/单位/缺口）——
   `pytest tests/test_io_contract_map.py` 保证 map ↔ profiles 双向同步；
2. **结构校验**：`odd-judge validate --profile <name>`（schema：warning_codes、
   missing_policy、阈值/枚举/置信度强校验）；
3. **缺信息语义**：profile `missing_policy`（`by_element → by_tag → default`，
   命中来源记入 `policy_source`）+ `definition_mode` 回退，对应 xlsx 全部「未决」表述。

配套单位换算约定（映射层吸收，感知侧无需改口径）：

| 量 | xlsx 口径 | 判据 observation_key 口径 | 换算 |
|---|---|---|---|
| 车速 | m/s | `vehicle.speed_kmh` | ×3.6 |
| 曲率 | 1/m | `scenery.curve_radius_m` | 1/\|κ\|，直道 clamp |
| 坡度 | rad | `*.pct` | tan×100 |
| 能见度 | 4 档 | `environment.visibility_m` | 代表值表（visibility_env.json） |

## 5. 关联文档

- [`docs/architecture_contract.md`](architecture_contract.md) §4 — 缺信息处理优先级与
  `missing_policy` 完整语义
- [`docs/mapping.md`](mapping.md) — 映射表（weather/visibility/friction/dynamic_roles…）
- [`docs/warning_codes.md`](warning_codes.md) — 判决码/告警码（ISMR 归因用）
- `judge/`（上游仓 ODD_Judge）`docs/semantics.md` — 三态判决语义
