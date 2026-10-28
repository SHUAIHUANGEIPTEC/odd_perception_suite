# 训练代码出处

本套件**只含运行时推理与判决**，不包含任何训练代码、数据审计工具或实验脚本。
训练发生在以下源仓库，需要复现实验时直接到对应位置：

## ODD 属性分类器（五任务）

- 源仓库：`/home/shuai/projects/codex_workspace/odd_multitask_yolo`
- 训练入口：`train.py --config configs/pilot_a_bf16.json`
- 选型/最终实验：`run_comparison.py` → `finalize_experiment.py`
  （验证集选择指标 = weather/lighting/road_condition/road_material 平均 macro-F1；
  测试集只用于最终一次性评测）
- 入仓权重：`runs/final_selected_e10/best.pth`（epoch 8）→ `weights/attr_best.pth`
- 数据：外部 CSV splits（RoadSaW/BDD100K/DENSE/SODA10M/ACDC/DAWN），
  `provenance.json` 记录哈希；不随本仓库分发
- 工具（未迁入）：`label_audit/`（标注审计 web 工具）、`sequence_video/`（视频叠加演示）

## Sparse4D 3D 检测器

- 源仓库：`/home/shuai/projects/dinov3_sparse4d_temporal_fixed`
- 唯一实验：`dinov3-official-vitb-sta-sparse4d-temporal-fixed-1024x576-fpn4_nusc-bboxes_1x4x8x1_4e-4_80000`
- 训练入口：`python run.py train`（4×A800，80k steps）
- 入仓权重：`last.ckpt`（iter-80000）→ `weights/sparse4d_last.ckpt`（软链）
- 运行要求见 `vendor/sparse4d/VENDOR.md`

## ODD 判别

规则引擎无需训练；规则 profile 的出处（BYD TJP ODC 表 B.1–B.3、
ISO 34503:2023、GB/T 43267、GB/T 45312）见 `THIRD_PARTY.md`。
