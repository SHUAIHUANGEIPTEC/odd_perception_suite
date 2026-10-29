# odd_perception_suite

统一 ODD 感知套件：**通用感知（Sparse4D 3D 检测）+ ODD 感知检出（环境属性分类）
+ ODD 判别（规则引擎三态判决）**，由三个独立项目整合为单一 monorepo。

```
                         ┌──────────────────────────────────────────────┐
  6× 相机图像 ──────────►│ 通用感知：DINOv3+STA/FPN4+Sparse4D 时序 3D 检测 │──► 3D 框
  (nuScenes 序列)        │ vendor/sparse4d + adapters/detect_adapter     │
                         └──────────────────────────────────────────────┘
                                             │ dynamic.* / sensors.*（汇总）
                         ┌───────────────────▼──────────────────────────┐
  单帧 RGB 图像 ────────►│ ODD 感知检出：YOLOv8n 五任务环境属性分类        │──► weather/
  (任意图片)             │ odd_suite/vendor/attr + adapters/attr_adapter │    visibility/
                         └───────────────────┬──────────────────────────┘    lighting/…
                                             │ environment.*（映射表）
                         ┌───────────────────▼──────────────────────────┐
  侧车 JSON（地图/CAN）─►│ Observation 组装（mapping/builder + sidecar）   │
                         └───────────────────┬──────────────────────────┘
                         ┌───────────────────▼──────────────────────────┐
                         │ ODD 判别：odd_judge 规则引擎（42 条 ODC 规则） │──► INSIDE /
                         │ profiles/byd_tjp.json（ISO 34503 语义）        │    BOUNDARY /
                         └──────────────────────────────────────────────┘    OUTSIDE
```

## 快速开始（零依赖，无需 GPU）

```bash
cd /home/shuai/projects/odd_perception_suite
pip install -e ".[dev]"        # 核心零第三方依赖；[dev] 只装 pytest

# 1) 全链路 mock 演示：三态（域内/临近边界/越界）一次看全
odd-suite demo

# 2) 单观测判决
odd-suite judge --obs examples/observations/sample_curve_boundary.json
odd-suite judge --scenario rain_outside --format json

# 3) 离线检测 JSON -> dynamic 汇总 -> 判决（无需 GPU/nuScenes）
odd-suite run --no-attr \
  --detections-json examples/detections/detections_sample.json \
  --sample-token demo-sample-0002 \
  --sidecar examples/sidecar_highway.json

# 4) 配置校验 / 权重校验
odd-suite validate --profile byd_tjp
odd-suite weights-check
```

## 可选：真实模型推理

```bash
# ODD 属性检出（CPU 可跑，需 torch）
pip install -e ".[attr]"
python scripts/make_demo_image.py    # 生成演示图（二进制不入库）
odd-suite attr examples/images/demo.png --checkpoint weights/attr_best.pth --device cpu

# 组合流水线：图像属性 + 侧车 + 判决
odd-suite run --images examples/images/demo.png \
  --sidecar examples/sidecar_highway.json --device cpu

# 真实 Sparse4D 推理（需 nuScenes + CUDA 算子，见 docs/env_setup.md）
pip install -e ".[detect]"
bash vendor/sparse4d/scripts/build_ops.sh
odd-suite run --detect-ckpt weights/sparse4d_last.ckpt \
  --data-root "$NUSCENES_DATA_DIR" \
  --sidecar examples/sidecar_highway.json
```

## 测试

```bash
# 核心用例（judge / mapping / pipeline / CLI，零第三方依赖，跳过 smoke_attr）
pytest -q tests/ -k "not smoke_attr"

# 全量（含 test_smoke_attr：真权重加载 + 224 前向，需 attr 环境，CPU torch 即可）
pip install -e ".[attr,dev]"        # torch / torchvision / numpy / pillow + pytest
pytest -q tests/
```

`test_smoke_attr` 读取 `weights/attr_best.pth`（缺失时自动 skip）。
三套环境配方（core / attr / detect）与依赖冲突说明见 `docs/env_setup.md`。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `odd_suite/` | 第一方胶水：CLI、schema、映射层、适配器、流水线、报告；`vendor/attr/` 为属性分类器模型代码与 checkpoint 契约 |
| `odd_judge/` | ODD 判别规则引擎（原 ODD_Judge 包原样迁入，零依赖，API 不变） |
| `vendor/sparse4d/` | Sparse4D 检测运行时（原交付包，内部导入不变；见 `VENDOR.md`） |
| `configs/ profiles/ switches/` | 流水线默认配置、ODD 规则 profile（BYD TJP / GB-T 骨架）、开关策略 |
| `weights/` | 最优权重各一份：`attr_best.pth`（拷贝）+ `sparse4d_last.ckpt`（软链）；`MANIFEST.json` 记录来源/哈希/指标 |
| `examples/` | 三态样例观测、侧车 JSON、检测样例、演示图 |
| `tests/` | 映射/组装/端到端/CLI 测试（核心用例无需 torch） |
| `docs/` | 架构、字段映射表、适配器扩展、环境安装、训练出处 |

## 关键集成逻辑

模型输出 → `Observation`（`odd_judge` 的统一输入契约）→ 三态判决：

| Observation 字段 | 来源（优先级：侧车 > 显式 > 映射默认） |
|---|---|
| `environment.*` | ODD 属性分类器 + 映射表（`odd_suite/mapping/tables/`，JSON 可配） |
| `dynamic.*` | Sparse4D 检测汇总（类别→角色映射可配：VRU / 两轮车 / 静态障碍 / 车流） |
| `sensors.*` | 检测分数均值 → 置信度/退化标志（策略 JSON 可配） |
| `scenery.*` / `vehicle.*` | 侧车 JSON / mock；CAN、地图适配器为文档化扩展点 |

- 判决语义：`INSIDE` → continue；`BOUNDARY`（临近边界带）→ warn_driver；
  `OUTSIDE` → mrm_or_refuse（BYD TJP profile，`definition_mode=restrictive`：缺失观测按域外处理）。
- 只有"最优"权重入仓：属性分类器为验证选型后的最终 ckpt（epoch 8，val 0.7775）；
  检测器为交付包内唯一 ckpt（iter-80000）。详见 `MANIFEST.json`。

## 文档

- `docs/architecture.md` — 分层与数据流
- `docs/mapping.md` — 完整字段映射表与优先级、扩展方式
- `docs/adapters.md` — 编写新适配器（CAN/地图/其他检测器）
- `docs/env_setup.md` — 三套环境配方（core / attr / detect）
- `docs/semantics.md` — ODD 判决语义（ISO 34503）
- `docs/warning_codes.md` — 告警码参考（命名规范、自动码表、引擎码）
- `docs/training/README.md` — 训练代码出处（不在本仓）
- `vendor/sparse4d/VENDOR.md` — 检测运行时来源与运行要求

## 代码风格与仓库卫生

- **C/C++/CUDA**：根目录 `.clang-format`（LLVM 基线，4 空格缩进，100 列）控制风格；
  `bash scripts/format_native.sh` 对第一方原生代码执行 clang-format。
  `vendor/sparse4d/` 是原样交付的第三方代码，**不**重排格式。
- **Python**：`pyproject.toml` 中的 ruff 配置（100 列、py310 目标、E/F/I/W 规则）；
  `vendor/` 与 `odd_judge/` 保持原样排除在外。
- **`.gitignore`**：排除权重二进制（`MANIFEST.json` 除外）、`__pycache__`、虚拟环境、
  运行时输出、编译产物（`*.so`/build）等；权重用 `odd-suite weights-check` 校验。

## 许可

本套件代码 Apache-2.0；第三方组件（MMYOLO 架构、DINOv3、ODD 规则标准引用）
见 `THIRD_PARTY.md` 与 `docs/licenses/`。
