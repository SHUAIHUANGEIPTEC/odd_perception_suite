# Vendored Sparse4D detection runtime

Source: `/home/shuai/projects/dinov3_sparse4d_temporal_fixed`
(a single-experiment delivery package: **DINOv3 ViT-B/16 + STA/FPN4 + Sparse4D
temporal 3D detection** on nuScenes, 6 cameras, 1024x576, temporal batch=1).

## What is vendored

| Path | Purpose |
|---|---|
| `bev_vae/` | model definitions (`models/perception/`), data pipeline, evaluation helpers, compiled-op source (`models/ops/deformable_aggregation/`) |
| `third_party/dinov3/` | trimmed official DINOv3 source (license: `docs/licenses/LICENSE-DINOV3.md`) |
| `configs/` | Hydra tree incl. the single experiment YAML |
| `scripts/build_ops.sh` | builds the Sparse4D deformable-aggregation CUDA op |
| `tools/` | `check_sparse4d_cuda_op.py`, `generate_sparse4d_anchors.py`, `smoke_handoff.py` |
| `det_dinov3_eval.py` | inference + nuScenes submission JSON builder |
| `nuscnes_eval.py` | metric computation (optional `eval-json` path) |
| `run.py` | vendor CLI (`train` / `infer` / `eval` / `eval-json`) |
| `nuscenes_kmeans900_bevvae.npy` | 900 Sparse4D anchors (39 KB) |
| `requirements.txt` | detection-stack pins (torch 2.5.1 / torchvision 0.20.1 / ...) |

## What was trimmed (not vendored)

- `test_sparse4d_*.py`, `requirements-test.txt` — package tests (left in the source repo)
- `README.md`, `VALIDATION.md`, `FILES.sha256`, `SOURCE_MANIFEST.json`,
  `CONFIG_EQUIVALENCE.json` — delivery metadata; key points summarized here
- `last.ckpt` (1.9 GB) — symlinked as `weights/sparse4d_last.ckpt`, see `MANIFEST.json`

Internal imports (`bev_vae.*`, Hydra config resolution) are unchanged; the
suite only touches this tree through `odd_suite/adapters/detect_adapter.py`
(`Sparse4DDetector`), which executes `run.py infer` as a subprocess inside this
directory.

## Runtime requirements (from the source delivery)

1. Linux + NVIDIA CUDA GPU (target: 4x A800 for training; inference works on one GPU).
2. Python 3.10 env with `requirements.txt` pins, then build the CUDA op:
   ```bash
   bash scripts/build_ops.sh    # TORCH_CUDA_ARCH_LIST=8.0 for A800; set for your GPU
   ```
3. nuScenes v1.0-trainval data **plus** the custom preprocessing caches:
   ```
   NUSCENES_DATA_DIR/
   ├── samples/ sweeps/ maps/ v1.0-trainval/
   └── nusc/
       ├── sensor_cache.feather, synchronization_cache.feather
       ├── scene2frame.json, scene2sensor2stamp2token.json
       ├── scene2sensor2intrinsic.json, scene2sensor2extrinsic.json
       ├── scene2stamp2annotation.json, scene2stamp2velocity.json
       ├── token2ego.json, token2file.json
   ```
4. Inference is scene/time-ordered with batch=1; do not shuffle frames or treat
   frames as independent images. Output is nuScenes submission JSON in global
   coordinates (10 classes: barrier, bicycle, bus, car, construction_vehicle,
   motorcycle, pedestrian, traffic_cone, trailer, truck; max 300 boxes; BF16).

The checkpoint `weights/sparse4d_last.ckpt` is the only one in the delivery
(iter-80000 final snapshot). It is loaded with strict full state-dict restore;
the DINOv3 pretrain file is **not** needed for inference (`--no-pretrained-backbone`).
