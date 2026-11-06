# ODD 判据维度 × 主机厂覆盖度

本文回答一个问题：**「ODD 判定维度映射表」里的每一维，行业里谁做、谁不做、做到什么程度。**

配套机器可读数据：[`coverage/oem_cn_l2l3.json`](../coverage/oem_cn_l2l3.json)（17 条维度，含数据源、覆盖层、阈值性质、置信度）。本文与 JSON 同源：JSON 是数据，本文是解释。**本文不产生判决规则**，`profiles/` 才是规则；维度与 `byd_tjp` 元素的对应关系写在 JSON 的 `profile_elements` 里。

口径：中国量产 L2+/L3 参照系（理想 / 小鹏 / 华为 ADS / 蔚来 / 比亚迪 TJP），依据是用户手册与功能说明书的 ODD 章节、GB/T 45312 与 GB/T 43267 的准入要求、ISO 34503 的声明格式。不含 Robotaxi 与国际 OEM 对比。所有对具体厂商行为的概括都带 `confidence`，`low` 的结论只当假设用，不要写进对外材料。

## 1. 四档定义

| 档位 | 定义 | 判定依据 |
|---|---|---|
| `mandatory` 法规强制 | 准入文件逐条要求，量产功能的激活/退出必须依赖 | 强标条文可直接引用，阈值不可协商 |
| `mainstream` 量产主流 | 无逐条强标，但不做会在常见场景出事或明显影响可用性 | 看量产功能是否真依赖它做降级决策 |
| `leading` 少数领先 | 自研 Tier1、城市 NOA 或 L3 安全论证才做 | 多数厂商只把它写进手册，不上线判据 |
| `rare` 罕见 | 基本不做；量产的等价物是简化监测，而非该维度本身 | 找不到公开的量产判据形态 |

## 2. 三层覆盖递减（本文最重要的结论）

「做」有三种深浅，行业里绝大多数厂商只到前两层：

| 层 | 含义 | 本文口径的维度数 |
|---|---|---|
| `declared` 声明层 | 写进用户手册 / 功能说明书 / 安全论证 | 17 |
| `online_judged` 在线判据层 | 运行时存在判据，能触发降级、接管提示或 MRM | 12（其中 `ood_confidence` 是 gate 而非维度，按独立维度口径为 11） |
| — 稳定触发降级 | 真正会造成功能降级或退出请求的判据 | 8 |
| `closed_loop_verified` 闭环可验证 | 判据有可量测的感知输入 + 定量阈值 + 可审计记录 | 6 |

**「手册 16 维，系统 8 维」是行业常态。** 看到一张 ODD 表逐行都配了感知输出，先问一句：这一行的判据在线跑吗？会触发降级吗？有记录可审计吗？三个问题能筛掉大半。

`stable_degradation` 没有对应 `layer_enum`，因为它不是一种「层」，而是 `online_judged` 的强度标注：12 条在线判据里，有 4 条只做提示或 gate（道路曲率、能见度、限速、OOD 可信度），不会单独触发降级。信号灯状态与交通密度连在线判据层都没到，只存在于声明层。

## 3. 主表

列说明：**实际数据源** 是本表相对原映射表的主要修正——原表把每一维都写成「感知输出数据项」，实际有四类维度并不来自环境感知。**阈值** 区分法定值与工程余量，前者改不了、后者可协商。

| 维度 | 档 | 原表标注的感知项 | 实际数据源 | 覆盖层 | 阈值 | 主机厂做法 | 置信度 |
|---|---|---|---|---|---|---|---|
| 道路类型 | mandatory | `out/map_topology`, `out/lane_geometry` | map_adas | 声明+在线 | regulation | 全员做，判据来自地图匹配置信度而非感知分类；只有无图/纯视觉路线用视觉补位 | high |
| 道路曲率 | mainstream | `out/lane_geometry.curvature` | map_adas（代理：车速×横摆角速度） | 声明+在线+闭环 | oem_declared | 做，但大量是运动学代理量；视觉车道曲率只有城市 NOA 玩家真做 | medium |
| 道路坡度 | leading | `out/lane_geometry.slope`, `out/ego_attitude.pitch` | chassis | 声明 | oem_declared | IMU/底盘 pitch 主要做控制前馈与交叉校验，很少升级为 ODD 判据 | medium |
| 车道宽度 | mainstream | `out/lane_geometry.width` | perception | 声明+在线+闭环 | oem_declared | 窄道/收费站/施工窄道触发降级；阈值多为 2.5 / 2.8 m 经验值 | high |
| 气象类型 | mainstream | `out/env_summary.weather` | perception（+ 雨刮档位） | 声明+在线+闭环 | oem_declared | 行业普遍只做雨/雪/雾三态；多标签与定量雨量极少 | high |
| 能见度 | mainstream | `out/env_summary.vis_bin` | perception | 声明+在线+闭环 | oem_declared | 档位判普遍且多为代理；用米做定量只在 L3 手册级声明里出现 | medium |
| 光照 | mandatory | `out/env_summary.illumination` | perception | 声明+在线+闭环 | oem_declared | 全员做；无照明夜间是最常见的功能边界 | high |
| 路面附着 | leading | `out/env_summary.road_cond + material` | inference | 声明 | oem_declared | 行业惯例是声明禁用（结冰/积雪/积水）而非量测 μ；量测 μ 基本没人做 | high |
| 交通参与者类型 | mandatory | `out/dyn_objects.type` | perception | 声明+在线+闭环 | oem_declared | 全员做（AEB/LCC 必需）；「未知」类型须判未决 | high |
| 交通密度 | leading | `out/traffic_density` | perception | 声明 | oem_declared | 在线定量少做，多用前车时距/拥堵等级代理；真做的是城市 NOA 博弈与 Robotaxi | medium |
| 速度 | mandatory | `out/ego_kinematics.vel` | chassis | 声明+在线 | regulation | 覆盖度最高，直接硬判，无感知依赖 | high |
| 地理围栏 | mandatory | `out/geofence` | map_adas | 声明+在线 | regulation | L4 叫 geofence；L2+ 以「开城白名单/已开放道路」等价形式存在 | high |
| 限速 | mandatory | `out/t_sr.value` | map_adas | 声明+在线 | regulation | 全员做（法规驱动），地图限速属性为主、视觉识标为辅 | high |
| 信号灯状态 | leading | `out/t_tl.state` | perception | 声明 | regulation | 纳入 ODD 判据的是少数；多数只当功能输入，出口靠路口类型 not_allowed 兜住 | medium |
| 驾乘人员状态 | mandatory | `out/driver_status` | dms | 声明+在线 | regulation | 全员做，但属独立 DMS 子系统，架构上归 ODC「人」要素 | high |
| 施工/异常事件 | mainstream | `event/anomaly` + `out/stat_objects` | perception | 声明（在线硬判 = leading） | oem_declared | 手册静态声明全员做；在线硬判仅 L3/L4；L2+ 主流只做「提示接管」 | medium |
| OOD 可信度 | rare | `out/idc`, `out/mobc`, `out/robc`, `out/low_quality_id` | perception | 声明+在线（gate） | oem_declared | 作为独立正交维度做成量产品的基本没有；等价物是传感器健康度/遮挡检测 | high |

原映射表末尾孤立的「道路施工 / 事故现场 / 交通管制」三行不是新维度，而是 `施工/异常事件` 的三条子事件，在 JSON 里映射到 `ODC.1.4.6 / 1.4.7 / 1.4.8`（profile 中本就是三条 `not_allowed` 元素）。

原表「限速/信号规则」一行在 JSON 里拆成 `speed_limit` + `signal_state` 两条：限速是法规强制、全员做，信号灯状态进 ODD 判据是少数，合成一行会把两档结论压平。故 JSON 为 17 条而非 16 条。

## 4. 三处需要修正的地方

**4.1 数据源归属错位。** 道路类型、地理围栏、限速三行标注为感知输出（`out/map_topology`、`out/geofence`、`out/t_sr.value`），实际来自 ADAS 地图 + 高精定位链路。判据系统的消费端必须知道这点：**感知退化不能兜住这三维**，地图失效时它们应当直接判未决或 OUT，而不是等感知给结论。同理，驾乘人员状态来自独立 DMS 子系统，OOD 可信度是感知链路的自评——两者都不属于「环境感知 ODD 维度」。

**4.2 阈值要分软硬。** 原表的「阈值/区间来源 = 模块 ODD 声明」把两类东西混在一起：

- `regulation`：法定或准入文件给出，例如车速上限、DMS 约束、限速规则。判据系统只做比对，不做放宽。
- `oem_declared`：厂商自定的工程余量，例如曲率半径下限、车道宽度下限、交通密度上限。可以协商、可以按项目覆写（`profiles/overrides/`）。

判据系统在报告里必须能区分「因为法定阈值越界」和「因为厂商余量越界」，否则客户无法判断责任的归属。

**4.3 OOD 可信度是 gate，不是第 17 维。** 原表备注写「正交于 ODD，用于防误判」，这个定位是对的，但现有 `definition_mode=restrictive` 只能表达「缺失 → OUT」，表达不了「存在但不可信 → 降级」。在判据系统里它的正确位置是**前置门**：先判感知是否可采信，采信了再判维度；不采信则整帧输出未决，而不是逐维输出一个不可信的 INSIDE。

## 5. 对照 `profiles/byd_tjp.json` 的缺口

把 17 条维度逐条对回 `byd_tjp` 的 42 条元素，最初有四个维度**没有对应元素**。其中三个已经补进扩展 profile `profiles/odc_ext_cn.json`（挂载方式 `--extends odc_ext_cn`）：

| 维度 | 生产者 | 元素 | 现状 |
|---|---|---|---|
| 路面附着 | `EnvMapper`：`road_condition`/`road_material` → `environment.friction_mu`（查表代理） | `ODC.1.5.1.road_friction` | ✅ 已补，真闭环（分类器本就有这两个头，原先只写 debug） |
| 交通密度 | `DynamicRollup`：traffic 角色计数 → `dynamic.surrounding_vehicles` | `ODC.4.2.1.traffic_density` | ✅ 已补，代理量（见下） |
| 限速 | `ObservationBuilder` 派生 `vehicle.overspeed_kmh`；限速本身来自地图侧车 | `ODC.2.1.5.speed_limit` | ✅ 规则已补，数据源仍受限于侧车，故带 `undefined_as: allowed` |
| 信号灯状态 | 地图/V2X SPaT → `scenery.traffic_light_state` + `traffic_light_applicable` | `ODC.1.5.1.traffic_light_state`、`ODC.1.5.2.intersection_annotation`（`--override city_noa_cn`） | ✅ 已补，城市 override 承载（见下） |

两点必须知道：

- **密度是代理不是密度。** nuScenes 检测框的 `translation` 是全局坐标，没有自车位姿就算不出「半径 50 m 内车辆数」，所以只统计画面内 traffic 角色目标数。元素阈值 15 是占位默认值，按场景覆写。
- **限速不能硬判。** 无地图适配器时缺 `speed_limit_kmh` 是常态，restrictive 模式下会把每一帧判成域外，因此该元素声明 `undefined_as: allowed`。这不是放松安全，而是承认「没有数据就不是域外」。

信号灯状态原先没有补，两个原因性质不同，现在都已处理：

- **语义冲突（已解）**：`byd_tjp` 把「平面交叉路口」判 `not_allowed`，信号灯规则在 TJP profile 里永远不可达。`profiles/overrides/city_noa_cn.json` 把路口禁令改为 `undefined`（城市场景下路口本身不再是域外条件），把约束下移为两条：信号灯状态合规、路口类型必标注（`signalized`/`unsignalized`/`none`，缺失即域外——这就是「无信号路口须标注」的可判形式）。
- **没有模型头（未解，但可选）**：本仓两个模型都不产出信号灯状态，`scenery.traffic_light_state` 目前只能来自地图/V2X SPaT。若要让视觉自给，正确做法是**独立 2D 信号灯检测头**（本地 BDD100K 就有 39,237 张带灯标注图，颜色分布 green 79k / none 57k / red 46k / yellow 3.4k），**不建议**给五任务分类器加第 6 个头：`vendor/attr/model.py` 的头是全局池化后的 `nn.Linear`，只能说「这一帧有没有灯、主要什么颜色」，路口 3~4 组灯会被平均掉；而且 `vendor/attr/checkpoint.py` 对 `classes` 做严格匹配 + `strict=True` 加载，加头等于重训并替换权重契约。

无论走哪条数据路径，`traffic_light_applicable`（这盏灯是否作用于本车道）都必须由车道关联给出，否则对向红灯会把本车判成域外。这层不在本仓实现。

反过来，`byd_tjp` 有而映射表未列的元素也不少（标线清晰、路口/环岛/匝道/收费站/铁路交叉、激活速度、横向/纵向加速度、眩光、ADS/DSSAD/DMS 系统状态、传感器退化与置信度），它们要么被并入上表的某一行，要么属于「系统健康」类而非 ODD 环境维度。

## 6. 怎么用这张表

- **做判据系统**：按 `actual_source` 分配数据来源与失效策略，不要假设感知能兜住地图类维度。
- **要现成规则**：`--extends odc_ext_cn` 挂上路面附着 / 交通流密度代理 / 限速合规三条；城市场景再叠 `--override city_noa_cn`（见第 5 节）。阈值是占位默认值，投产前按项目覆写到 `profiles/overrides/`。
- **做客户方案**：用 `tier` 剪枝。客户是量产 L2+ 时，`leading` 与 `rare` 档维度可以只做声明不做在线判据；客户要 L3 准入时，`declared` 层的 17 条一条都不能少。
- **看竞品**：先看它 `online_judged` 有几条，而不是手册里的 OD 声明有多长。

## 7. 免责

档位是对**厂商群像**的概括，不是对某一车型或某一版本的断言。`confidence` 为 `medium`/`low` 的条目依据是产品形态与公开说明的推断，落地前需以目标客户的实际 ODD 文档为准。JSON 的 `as_of` 记录推断时点，行业共识变化后应整体复核。
