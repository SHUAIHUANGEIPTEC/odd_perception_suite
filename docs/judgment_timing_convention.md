# 判决节拍与观测时效约定（DRAFT — 待感知侧联席确认）

> **状态**：草案 v0.1（2026-11-06 起草），对应
> [`io_contract_review.md`](io_contract_review.md) 决策项 **C4**。
> 本文在联席确认前仅作会议底稿；确认后去掉 DRAFT 标记转正，
> 并按 §6 清单落实实现点。数值均为**拟议值**，以联席结论为准。

## 1. 判决节拍

| 模式 | 节拍 | 说明 |
|---|---|---|
| 离线回放（现状） | **逐观测帧** | JSON 单帧 / JSONL 逐行判决（`OddJudge.evaluate_path` / `evaluate_stream`），一帧一条 `JudgmentReport` |
| 在线（拟议） | **10 Hz** | 与融合感知出帧同频；1 Hz 的环境/地图字段在有效期内（§3）跨判决帧复用 |

## 2. 时间戳约定

- `Observation.timestamp`：**ISO8601 UTC**（与 xlsx「时间戳与单位约定」一致），
  是该判决帧的对齐基准；epoch 毫秒等价可接受。
- **字段级时间戳**：各 `out/*` 记录携带自身 `timestamp`（`out/dyn_objects` 已含；
  其余字段经 `out/provenance` 或记录内时间携带——承载方式待联席确认）。
- 全链路同钟（UTC），容许偏差拟议 ≤ 50 ms；GNSS 失锁时以 DR 标志为准（xlsx 输入表已有）。

## 3. 字段组有效期（max_age，拟议）

过期判定：`age = Observation.timestamp − 字段时间戳`，`age > max_age` 即过期。
取值原则 = **源周期 × 5**（抗抖动，允许丢 4 帧），初值如下：

| 字段组 | 源频率（xlsx） | 拟议 max_age | 过期动作 |
|---|---|---|---|
| `environment.*`（`out/env_summary`） | 1 Hz | **2.0 s** | 丢弃该键 → 缺失回退 |
| `scenery.*` 地图类（`out/map_topology` / `geofence` / `lane_geometry`） | 1 Hz | **3.0 s** | 同上 |
| `dynamic.*`（`out/dyn_objects`） | 10 Hz | **0.5 s** | 同上 |
| `dynamic.surrounding_vehicles`（若源自 `out/traffic_density`） | 1 Hz | **2.0 s** | 同上（例外单列） |
| `vehicle.*` / `sensors.*`（ego 50~100 Hz、`out/robc` 10 Hz） | 10~100 Hz | **0.3 s** | 同上 |
| `vehicle.driver_*` / DMS（`out/driver_status`） | 2 Hz | **1.5 s** | 同上 |
| 限速/信号（`out/t_sr` / `out/t_tl`） | 10 Hz | **0.5 s** | 同上 |

## 4. 过期处理机制（复用现有语义，引擎零改动）

1. **装配层过滤**：adapter/builder 写入字段前检查 §3，过期则**不写入该键**
   （`ObservationBuilder.add` 对缺席键天然不产出）。
2. **键缺失 → 引擎既有回退**：`missing_policy`（`by_element → by_tag → default`）→
   `UNDEFINED`（`MISSING-POLICY-UNDEFINED`，不参与聚合），未命中策略则
   `definition_mode`（byd_tjp = restrictive → OUTSIDE）。
3. **不引入新的引擎状态/字段**：时效语义完全由「缺键」表达，
   与 C1–C3 已落地的机制同构。

> ⚠️ 装配层的 staleness 过滤**尚未实现**（功能新增），联席后排期；
> 在此之前超期观测会原样进入判决，是已知缺口。

## 5. 乱序 / 丢包 / 冻结源

- **未来戳**（字段时间戳 > 帧时间戳，容限 50 ms 外）：视为时钟异常，按缺失处理并记录告警。
- **丢包**：不重传；过期即缺失，交由 §4 回退。
- **冻结源**（源停更但心跳仍在）：最后有效值保持至 max_age，超时按缺失。
- **乱序帧**（帧时间戳回退）：判决照常，仅在报告 `timestamp` 中可见；不倒带重判。

## 6. 联席确认清单（与感知侧逐项过）

- [ ] §3 max_age 数值（当前为源周期 × 5 拟议值）
- [ ] 时钟源与时钟同步要求（≤ 50 ms 是否可达；GNSS 失锁降级策略）
- [ ] 字段级 timestamp 承载方式：`out/*` 记录内 vs `out/provenance` 旁路
- [ ] 在线模式 10 Hz 节拍与感知出帧一致性（是否存在背压/降频）
- [ ] staleness 过滤实现点归属（suite 装配层，功能新增，排期）
- [ ] 过期降级的对外口径：是否需要在 `JudgmentReport` 标注
      「本次判决含过期字段回退」（现可用 `undefined_handled[].policy` 间接观测）

## 7. 关联文档

- [`io_contract_review.md`](io_contract_review.md) — 接口评审与 C1–C4 决策状态
- [`architecture_contract.md`](architecture_contract.md) §4 — 缺信息处理优先级
- [`mapping.md`](mapping.md) — 字段映射与装配层
- `judge/`（上游仓 ODD_Judge）`docs/semantics.md` — 三态判决与缺失回退语义
- xlsx《感知输入输出数据定义》「说明」页 — 频率/时延/坐标系原始约定
