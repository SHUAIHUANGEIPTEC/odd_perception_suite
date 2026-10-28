# Third-Party Notices

This project consolidates code and weights from three source projects. Provenance
and license notes for each are recorded here and in `MANIFEST.json`.

## 1. ODD attribute classifier (from `odd_multitask_yolo`)

- Source: `/home/shuai/projects/codex_workspace/odd_multitask_yolo`
- Vendored at: `odd_suite/vendor/attr/` (`model.py`, `data.py`)
- Architecture: YOLOv8n CSPDarknet P5 backbone adapted from
  [OpenMMLab MMYOLO](https://github.com/open-mmlab/mmyolo), **Apache-2.0**
  (original copyright OpenMMLab). Checkpoint and module names follow the local
  MMYOLO `YOLOv8CSPDarknet` / `CSPLayerWithTwoConv` / `SPPFBottleneck`
  implementations for weight-key compatibility. Global pooling and the five
  classification heads are new.
- Note: the source repo's `LICENSE-MMYOLO` file contains the **GPLv3 text**,
  which contradicts its own `THIRD_PARTY.md` declaration (Apache-2.0) and the
  upstream MMYOLO license. We preserve that file verbatim as
  `docs/licenses/LICENSE-MMYOLO` for provenance; the intended upstream license
  is Apache-2.0.

## 2. Sparse4D 3D detection runtime (from `dinov3_sparse4d_temporal_fixed`)

- Source: `/home/shuai/projects/dinov3_sparse4d_temporal_fixed`
- Vendored at: `vendor/sparse4d/`
- Contains a trimmed copy of the official **DINOv3** source
  (`vendor/sparse4d/third_party/dinov3/`), see
  `docs/licenses/LICENSE-DINOV3.md`. Do not run upstream DINOv3 training
  commands on this trimmed tree.
- Detection model internals (Sparse4D head, temporal memory, deformable
  aggregation CUDA ops, nuScenes dataset code) are kept as delivered,
  unchanged.

## 3. ODD judgment engine (from `ODD_Judge`)

- Source: `/home/shuai/projects/ODD_Judge`
- Vendored at: `odd_judge/` (unchanged, zero runtime dependencies)
- Rule profiles reference: BYD TJP ODC Tables B.1/B.2/B.3 (traffic-jam pilot
  design operating conditions), ISO 34503:2023, GB/T 43267-2023,
  GB/T 45312 (ODC element hierarchy). These standards are referenced, not
  redistributed.

## 4. Weights

See `MANIFEST.json` for source paths, sha256 checksums, metrics and selection
rationale of the two shipped checkpoints.
