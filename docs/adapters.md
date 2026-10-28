# 适配器与扩展

## ObservationAdapter 契约

沿用 `odd_judge.perception.base.ObservationAdapter`：

```python
class ObservationAdapter(ABC):
    def get_observation(self) -> Observation: ...
    def get_stream(self) -> list[Observation]:  # 默认单帧
        return [self.get_observation()]
```

新增感知源时实现该 ABC，返回带字段组的 `Observation` 即可接入流水线。

## 现有适配器

| 适配器 | 输入 | 输出字段组 |
|---|---|---|
| `ImageAttrAdapter` | 图像路径 + `weights/attr_best.pth` | `environment` |
| `NuScenesDetAdapter` | `list[Detection3D]` | `dynamic` + `sensors` |
| `JsonSidecarAdapter` | 侧车 JSON | `scenery` / `vehicle`（及任何覆写） |
| `DemoScenario`（mock） | 场景名 | 合成 attr + 检测 + 完整观测 |

## 接入新感知源（示例：CAN / 地图）

```python
from odd_judge.core.types import Observation
from odd_judge.perception.base import ObservationAdapter

class CanMapAdapter(ObservationAdapter):
    def __init__(self, can_client, map_service):
        self.can, self.map = can_client, map_service

    def get_observation(self) -> Observation:
        return Observation(fields={
            "vehicle": {
                "speed_kmh": self.can.speed_kmh,
                "seatbelt_on": self.can.seatbelt_on,
                "driver_in_sight": ...,
            },
            "scenery": {
                "road_type": self.map.current_road_type(),
                "curve_radius_m": self.map.curve_radius_ahead(),
                ...
            },
        }, source="can_map")
```

然后并入 `Pipeline.run(sidecar=...)` 或直接调用
`Pipeline.run(..., sidecar=adapter.get_observation().fields)`；
也可在 `odd_suite/pipeline/runner.py` 增加一个可配置的 sidecar 适配器插槽。

## 接入其他 3D 检测器

实现 `odd_suite/adapters/base.py` 的 `Detector3D` 协议：

```python
class Detector3D(Protocol):
    def detect(self, *, sample_token: str | None = None) -> list[Detection3D]: ...
```

返回任意来源的 `Detection3D` 列表后，`DynamicRollup` / `SensorDefaults`
的汇总逻辑不变；新类别只需在 `dynamic_roles.json` 或
`MapperRegistry.register_role()` 登记角色。

## 接入其他属性模型

`AttrModel` 协议只需 `predict(image: Path) -> AttrResult`。
`AttrResult.prediction` 必须是 `{task: value}`（weather 为 list，其余为 str），
任务集合与 `odd_suite/vendor/attr/classes.py::CLASSES` 对齐即可复用映射表。

## 新增 ODD 规则

规则引擎部分（ODC 元素、规则类型）见 `odd_judge/config/registry.py`
与 `odd_judge/cli.py validate`；profile 增加元素后无需改本套件代码。
