# 字段映射（模型输出 → ODD Observation）

映射表全部在 `odd_suite/mapping/tables/*.json`，可整体覆写（`--mapping-dir` / `PipelineConfig.mapping_dir`）。
映射层为纯 Python，无 torch。

## 优先级

```
侧车 JSON 显式值  >  映射默认值  >  缺失（由 odd_judge 的 definition_mode 处理）
```

（`definition_mode=restrictive` 时缺失 → OUTSIDE，安全优先；因此"glare"这类无模型头的
字段应由侧车提供。）

## environment（来自 ODD 属性分类器）

| 分类输出 | 写入字段 | 默认值 | 说明 |
|---|---|---|---|
| baseline | `rain_mm_per_h` / `snow_level` / `fog` | 0.0 / "none" / false | 先写基线，多标签各自覆盖 |
| weather ∋ Rain | `rain_mm_per_h` | 4.0 | ≤14.9 阈值内 |
| weather ∋ Snow | `snow_level` | **"moderate"** | 分类器无强度头，默认保守 → BOUNDARY 预警；表内可改 "light" |
| weather ∋ Fog | `fog` | true | 且 `visibility_m` 钳到 ≤350 |
| weather ∋ Haze / Dust | 仅钳位 | `visibility_m` ≤200 | 轻度衰减保守处理 |
| visibility = LT50 / 50_200 / 200_500 / GE500 | `visibility_m` | 40 / 120 / 350 / 2000 | 档位保守代表值；GE500 是上限非测量 |
| lighting = Day / NightLit / DawnDusk / NightUnlit | `illumination_lux` | 5000 / 150 / 80 / 20 | |
| 无预测头 | `glare` | 缺失 | 仅侧车可提供 |
| road_condition ∈ Dry / Wet / Snow-covered | `friction_mu` | 0.85 / 0.6 / 0.3 | 查表代理，非量测；见 `friction_env.json` |
| road_material ∈ Concrete / Cobblestone | `friction_mu` 取上限 | 0.8 / 0.55 | `min(condition_mu, material_cap)`，石板路干态也滑 |
| road_condition = Other 或缺失 | 不写 `friction_mu` | — | 未知表面留空 → 由 `odc_ext_cn` 判未决 |

`road_condition` / `road_material` 仍同时写入 `PipelineResult.debug["road"]`。

## dynamic（来自 3D 检测汇总，`dynamic_roles.json`，score 阈值默认 0.0 / CLI `--score-threshold`）

| detection_name | 角色 | 目标字段 |
|---|---|---|
| pedestrian | vru | `pedestrians_present = count>0` |
| bicycle, motorcycle | two_wheeler | `two_wheelers_present = count>0` |
| barrier, traffic_cone | obstacle | `obstacles_over_0_3m = count` |
| car, truck, bus, trailer, construction_vehicle | traffic | `surrounding_vehicles`（计数，密度代理） |
| animal（nuScenes 无此类） | animal | `animals_over_0_3m = 0`（扩展钩子） |
| 未登记类 | ignore（可配） | — |

为何车流不计 obstacles：`ODC.4.1.4.obstacle` 规则是 `<=0`，TJP 场景中周围车辆是常态；
把车流算成障碍会让任何有车环境直接 OUTSIDE。其他 profile 可把 traffic 角色改配为 obstacle。

`surrounding_vehicles` 是**密度代理**而非密度：nuScenes 检测框的 `translation` 是全局坐标，
没有自车位姿就算不出「半径 N 米内的车辆数」，所以这里只统计画面内 traffic 角色目标数，
由 `odc_ext_cn` 的 `ODC.4.2.1.traffic_density` 消费。

## sensors（`sensor_policies.json`）

| 条件 | 结果 |
|---|---|
| 有 ≥1 检测框 | `confidence_* = mean(detection_score)`；`*_degraded = conf < 0.3` |
| 无检测框 | 默认 camera 0.95 / lidar 0.92，degraded=false |
| 侧车提供 | 侧车覆盖 |

## scenery / vehicle

两个模型都不产出；来源优先级：侧车 JSON（`examples/sidecar_highway.json`）→ mock → 未来
`CanMapAdapter`（`docs/adapters.md`）。

城市场景（`--override city_noa_cn`）额外消费三个字段，同样只能来自地图 / V2X SPaT / 独立信号灯感知：

| 字段 | 取值 | 含义 |
|---|---|---|
| `scenery.traffic_light_state` | `green` / `yellow` / `red` / `unknown` | 本车道的信号灯状态；`unknown` 按 allowed 语义判域外 |
| `scenery.traffic_light_applicable` | `true` / `false` | 该灯是否作用于本车道；`false` 或缺失时信号灯规则跳过（不判域外） |
| `scenery.intersection_type` | `signalized` / `unsignalized` / `none` | 路口类型标注；缺失即域外（对应「无信号路口须标注」） |

`traffic_light_applicable` 是这组契约的关键：路口常有 3~4 组灯，只有与自车车道关联的那盏才该
参与判决，否则对向红灯会把本车判成域外。车道关联不在本仓实现，由地图/感知侧提供。

## 派生字段（`ObservationBuilder.build`）

判据引擎只能拿「一个观测字段 vs 一个阈值」比较，跨信号比较在合并阶段先算成派生量：

| 派生字段 | 公式 | 输入缺失时 |
|---|---|---|
| `vehicle.overspeed_kmh` | `speed_kmh - speed_limit_kmh` | 字段不产生（规则可声明 `undefined_as`） |

`speed_limit_kmh` 只能来自地图侧车（本仓无地图适配器），所以 `odc_ext_cn` 的限速元素带
`undefined_as: allowed`：缺侧车时跳过，而不是在 restrictive 模式下把每一帧判成域外。

## 扩展

```python
from odd_suite.mapping import MapperRegistry

# 1) 增加环境映射规则（如把 road_condition 映射到自定义 scenery 字段）
MapperRegistry.register_env_rule(
    "road_surface",
    lambda attr, env: {"scenery.surface_wet": attr.road_condition == "Wet"},
)

# 2) 改变某类检测的角色（如把 car 也算作障碍——仅适用于非 TJP profile）
MapperRegistry.register_role("car", "obstacle")
```

`MapperRegistry.load_tables()` 会自动应用角色覆写。
