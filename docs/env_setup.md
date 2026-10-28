# 环境安装（三套配方）

核心、属性检出、3D 检测的依赖差异大（尤其 torch 版本），**不要**强行共用一个环境。

## 1) odd-core（零第三方依赖）

```bash
cd /home/shuai/projects/odd_perception_suite
python -m venv .venv && . .venv/bin/activate     # 或 conda create -n odd-core python=3.12
pip install -e ".[dev]"
```

可用命令：`demo`、`judge`、`validate`、`list-rules`、`weights-check`、
`run --no-attr --detections-json ...`。测试门槛：

```bash
pytest -q tests/ -k "not smoke_attr"
```

## 2) odd-attr（ODD 属性分类，CPU 可用）

原训练环境 torch 2.7.0+cu128；CPU 推理任意 torch>=2.3 均可。

```bash
conda create -n odd-attr python=3.12 -y
conda activate odd-attr
pip install -e ".[attr,dev]"
# GPU 版按官方源安装对应 CUDA 构建，例如：
# pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

pytest -q tests/test_smoke_attr.py -v
python scripts/make_demo_image.py     # 生成 examples/images/demo.png（不入库）
odd-suite attr examples/images/demo.png --checkpoint weights/attr_best.pth --device cpu
```

## 3) odd-detect（Sparse4D 3D 检测，需 CUDA GPU + nuScenes）

按 `vendor/sparse4d/requirements.txt` 钉扎（torch 2.5.1 / torchvision 0.20.1 /
pytorch-lightning 2.1.0 / hydra 1.3.2 / nuscenes-devkit 1.2.0 ...）。
该钉扎与 odd-attr 的 2.7 冲突，请在独立环境中安装。

```bash
conda create -n odd-detect python=3.10 -y
conda activate odd-detect
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r vendor/sparse4d/requirements.txt
pip install -e ".[detect,dev]"
bash vendor/sparse4d/scripts/build_ops.sh        # TORCH_CUDA_ARCH_LIST 按你的 GPU 设置

export NUSCENES_DATA_DIR=/path/to/nuscenes        # 需含 nusc/* 预处理缓存，见 VENDOR.md
python vendor/sparse4d/tools/check_sparse4d_cuda_op.py
odd-suite run --detect-ckpt weights/sparse4d_last.ckpt \
  --data-root "$NUSCENES_DATA_DIR" --sidecar examples/sidecar_highway.json
```

## 同机双模型

用两个 venv/conda 环境分别跑 attr 与 detect；两者之间只靠
"检测预测 JSON 文件 + 属性 JSON" 交接（`run --detections-json` 即离线交接口），
无需同进程。若强行共用一个环境，请以 detect 的 torch 2.5.1 为基准重装并重建
CUDA 算子，且重跑 `tests/test_smoke_attr.py` 验证属性分支（未在本交付中验证该组合）。
