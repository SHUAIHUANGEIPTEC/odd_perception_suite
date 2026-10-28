"""CPU packaging smoke test: actual ViT-B/STA/FPN/head, two synthetic frames.

Run from the package root with: python -m tools.smoke_handoff
No pretrained weights, dataset, network download or CUDA compiler is needed.
This checks packaging and execution, not trained accuracy or CUDA correctness.
"""
import os
from pathlib import Path
import tempfile

import hydra
from hydra import compose, initialize_config_dir
import torch

from run import EXPERIMENT, ROOT


def main():
    os.environ['PROJECT_ROOT'] = str(ROOT)
    os.environ.setdefault('NUSCENES_DATA_DIR', str(ROOT / 'data'))
    torch.set_num_threads(2)
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['experiment=' + EXPERIMENT])
    assert cfg.devices == 4 and cfg.batch_size == 8 and cfg.max_steps == 80000
    assert cfg.trainer.precision == 'bf16-mixed'
    assert cfg.data.temporal_sampling and not cfg.trainer.use_distributed_sampler
    cfg.model.backbone.weight_path = None
    cfg.model.sparse4d_head.use_deformable_func = False
    model = hydra.utils.instantiate(cfg.model).eval()
    assert model.backbone.model.__class__.__module__.startswith('dinov3.')
    from det_dinov3_eval import predict_bboxes
    import nuscnes_eval
    import bev_vae.train
    assert nuscnes_eval.NuScenesEval_custom is not None
    images = torch.zeros(6, 3, 64, 96)
    metas = dict(cam_num=6, img_shape=(64, 96),
        intrinsic=torch.tensor([[48., 0., 48.], [0., 48., 32.], [0., 0., 1.]]).repeat(1, 6, 1, 1),
        extrinsic=torch.eye(4).repeat(1, 6, 1, 1), geometric=torch.eye(3).repeat(1, 6, 1, 1),
        ego_to_global=torch.eye(4, dtype=torch.float64)[None], sequence_id=['smoke'],
        temporal_scope='inference')
    with torch.no_grad():
        for frame in range(2):
            metas.update(timestamp_ns=torch.tensor([1530000000000000000 + frame * 500000000]),
                         sequence_start=torch.tensor([frame == 0]), frame_index=torch.tensor([frame]))
            boxes = predict_bboxes(model, images, metas, score_thr=0., max_num=10, precision='32', head='sparse4d')
            assert torch.isfinite(boxes).all() and boxes.shape == (1, 10, 12)
            context = model.sparse4d_head.temporal_memory.replays['inference']
            assert context['valid'].item() == bool(frame)
            assert abs(context['interval'].item() - .5) < 1e-6
    with tempfile.TemporaryDirectory() as tmp:
        cfg.paths.output_dir = tmp
        callbacks = [hydra.utils.instantiate(c) for c in cfg.callbacks.values()]
        logger = hydra.utils.instantiate(cfg.logger.tensorboard)
        logger.experiment.add_scalar('handoff/smoke', 1, 0)
        logger.finalize('success')
    print(f'PASS: full model ({sum(p.numel() for p in model.parameters()):,} parameters), '
          f'two 6-camera synthetic frames, recurrent state, {len(callbacks)} callbacks, logger and evaluator imports.')


if __name__ == '__main__':
    main()
