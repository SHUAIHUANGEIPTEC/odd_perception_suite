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
| road_condition / road_material | 不映射判决字段 | `PipelineResult.debug["road"]` | 扩展钩子 `register_env_rule` |

## dynamic（来自 3D 检测汇总，`dynamic_roles.json`，score 阈值默认 0.0 / CLI `--score-threshold`）

| detection_name | 角色 | 目标字段 |
|---|---|---|
| pedestrian | vru | `pedestrians_present = count>0` |
| bicycle, motorcycle | two_wheeler | `two_wheelers_present = count>0` |
| barrier, traffic_cone | obstacle | `obstacles_over_0_3m = count` |
| car, truck, bus, trailer, construction_vehicle | traffic | **不计入**任何判决字段（debug 统计） |
| animal（nuScenes 无此类） | animal | `animals_over_0_3m = 0`（扩展钩子） |
| 未登记类 | ignore（可配） | — |

为何车流不计 obstacles：`ODC.4.1.4.obstacle` 规则是 `<=0`，TJP 场景中周围车辆是常态；
把车流算成障碍会让任何有车环境直接 OUTSIDE。其他 profile 可把 traffic 角色改配为 obstacle。

## sensors（`sensor_policies.json`）

| 条件 | 结果 |
|---|---|
| 有 ≥1 检测框 | `confidence_* = mean(detection_score)`；`*_degraded = conf < 0.3` |
| 无检测框 | 默认 camera 0.95 / lidar 0.92，degraded=false |
| 侧车提供 | 侧车覆盖 |

## scenery / vehicle

两个模型都不产出；来源优先级：侧车 JSON（`examples/sidecar_highway.json`）→ mock → 未来
`CanMapAdapter`（`docs/adapters.md`）。

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
