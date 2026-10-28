"""Portable entry points for the one supported detection experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXPERIMENT = 'dinov3-official-vitb-sta-sparse4d-temporal-fixed-1024x576-fpn4_nusc-bboxes_1x4x8x1_4e-4_80000'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'infer', 'eval', 'eval-json'])
    args, remaining = parser.parse_known_args()
    os.chdir(ROOT)
    os.environ['PROJECT_ROOT'] = str(ROOT)
    os.environ.setdefault('OUTPUT_DIR', str(ROOT / 'outputs'))
    os.environ.setdefault('CKPT_DIR', str(ROOT / 'weights'))
    os.environ.setdefault('SPARSE4D_ANCHOR_PATH', str(ROOT / 'nuscenes_kmeans900_bevvae.npy'))
    os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ.setdefault('HYDRA_FULL_ERROR', '1')

    if args.command == 'train':
        cmd = [sys.executable, '-m', 'bev_vae.train', 'experiment=' + EXPERIMENT, *remaining]
    elif args.command in {'infer', 'eval'}:
        cmd = [sys.executable, str(ROOT / 'det_dinov3_eval.py'),
               '--experiment', EXPERIMENT, '--precision', 'bf16-mixed',
               '--no-pretrained-backbone']
        if args.command == 'infer':
            cmd.append('--no-run-nuscenes-eval')
        cmd.extend(remaining)
    else:
        sub = argparse.ArgumentParser(description='Evaluate an existing nuScenes detection prediction JSON.')
        sub.add_argument('--result-path', type=Path, required=True)
        sub.add_argument('--data-root', default=os.environ.get('NUSCENES_DATA_DIR'), required=not os.environ.get('NUSCENES_DATA_DIR'))
        sub.add_argument('--output-dir', type=Path, default=ROOT / 'eval_results')
        options = sub.parse_args(remaining)
        if not options.result_path.is_file():
            sub.error(f'Prediction JSON does not exist: {options.result_path}')
        from det_dinov3_eval import run_nuscenes_eval
        run_nuscenes_eval(options.data_root, options.result_path, options.output_dir)
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
