# ODD 判决语义

## 1. 三态分类（ISO 34503 Clause 6）

| 状态 | 含义 | 系统期望行为 |
|------|------|-------------|
| **INSIDE** | 满足 ODD | ADS 执行 DDT，性能达标 |
| **BOUNDARY** | 接近限制值 / Conditional 切换区 | 降级、限速、预警、准备 MRM |
| **OUTSIDE** | 不满足 ODD | 拒绝激活 / transition demand / MRM |

## 2. 元素要求映射

### allowed（允许）

观测值 → 直接映射：

| 观测状态 | 判决 |
|---|---|
| 在阈值内且有裕度 | INSIDE |
| 落入边界带 | BOUNDARY |
| 超出阈值 | OUTSIDE |

### not_allowed（不允许）

「禁止条件**出现**」是越界触发。语义反转：

| 禁止条件 | 判决 |
|---|---|
| 出现（如：有行人、有路口） | OUTSIDE |
| 临近（如：枚举 boundary_values） | BOUNDARY |
| 不存在 | INSIDE |

### undefined（不定义）

不评估，记入 `undefined_handled`，不参与聚合。

## 3. 定义模式（DefinitionMode）— 缺失观测回退

| 模式 | 行为 | 用途 |
|------|------|------|
| **permissive** | 视为 INSIDE，跳过 | 演示 / 宽松原型 |
| **restrictive** | 视为 OUTSIDE（`UNDEFINED-STRICT`） | **BYD TJP 默认**，安全优先 |
| **default** | UNDEFINED，不入聚合 | ODC 骨架探索 |

## 4. 定量边界（BoundaryMargin）

```
absolute: band = value
relative: band = |threshold| * value   (between: band = (hi-lo) * value)
```

判定（以 `>=` 为例）：
```
value >= thr + band   → INSIDE
thr <= value < thr+band → BOUNDARY   (remaining_margin = value - thr)
value < thr            → OUTSIDE    (remaining_margin < 0)
```

`remaining_margin`：到最近阈值的**有符号距离**（正=域内，负=越界）。

## 5. 定性边界

- **枚举**：`boundary_values` 列表显式标注边界取值
- **布尔**：二值无边界；可选配 `confidence_threshold` + `confidence_band`，置信度落入带内 → BOUNDARY

## 6. 组合条件（Conditional / AllOf / AnyOf / Not）

```json
{"all_of": [
  {"key": "scenery.road_type", "op": "in", "value": ["expressway_main"]},
  {"not": {"key": "scenery.construction_zone", "op": "==", "value": true}}
]}
```

`applies_when` 为假时规则跳过（`CONDITIONAL-SKIP`），不参与判决。

## 7. 聚合与建议

```
优先级: OUTSIDE > BOUNDARY > INSIDE > UNDEFINED
severity: OUTSIDE→critical, BOUNDARY→warning, INSIDE→info
recommendation: INSIDE→continue, BOUNDARY→warn_driver, OUTSIDE→mrm_or_refuse
```

## 8. 开关（Switches）

- `global_enabled=false`：全部跳过
- `by_element[id]=false`：按元素关闭
- `by_rule_type[type]=false`：按规则类型关闭
- `by_tag[tag]=false`：按标签关闭
- `boundary_warnings_enabled=false`：BOUNDARY 不输出预警，整体可降为 INSIDE

关闭的规则记入 `skipped[]`，不参与判决。
