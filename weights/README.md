# Weights

Only the best checkpoint of each model is kept (see `../MANIFEST.json` for
provenance, checksums, metrics and selection rationale).

| File | Size | Kind | Source |
|---|---|---|---|
| `attr_best.pth` | ~15 MB | ODD attribute classifier (epoch 8, val score 0.7775) | copied from `odd_multitask_yolo/runs/final_selected_e10/best.pth` |
| `sparse4d_last.ckpt` | ~1.9 GB | Sparse4D 3D detector (iter-80000 final snapshot) | **symlinked** to `dinov3_sparse4d_temporal_fixed/last.ckpt` |

## Re-linking the large checkpoint

If the repository or the source project moves, recreate the symlink:

```bash
ln -sfn /path/to/dinov3_sparse4d_temporal_fixed/last.ckpt \
  /path/to/odd_perception_suite/weights/sparse4d_last.ckpt

# verify
odd-suite weights-check
```

The detection checkpoint is not required for `odd-suite demo`, `odd-suite judge`
or `odd-suite attr`; it is only used by `odd-suite run --detect-ckpt ...`.
