from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from bev_vae.data.datasets.utils import read_json_file
from bev_vae.data.transforms import denormalize
from bev_vae.evaluation.perception import calc_det_metrics


class Score(Callback):
    def __init__(self, cache_size, epoch_interval=1, eval_det_cfg_path=None, save_dir=None):
        self.save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
        self.cache_size = cache_size
        self.epoch_interval = epoch_interval
        self.eval_det_cfg_path = eval_det_cfg_path
        self.eval_det_cfg = read_json_file(Path(eval_det_cfg_path)) if eval_det_cfg_path is not None else None
        self.act1 = []
        self.act2 = dict()
        self.psnr = dict()
        self.ssim = dict()
        self.mvsc = dict()
        self.pd_bboxes_list = []
        self.gt_bboxes_list = []
        self.det_logged = False

    @torch.no_grad()
    def calculate_activation_statistics(self, model, images):
        model.eval()
        act = model(images)[0].squeeze(3).squeeze(2).cpu().numpy()
        return act

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean.real)
            else:
                fid = np.float64(-1)
        else:
            fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return fid

    def calculate_psnr(self, targets, images):
        return np.mean([psnr(targets[i], images[i], data_range=1.0) for i in range(len(targets))])

    def calculate_ssim(self, targets, images):
        return np.mean([ssim(targets[i], images[i], data_range=1.0, channel_axis=0) for i in range(len(targets))])

    @torch.no_grad()
    def calculate_mvsc(self, model, images):
        model.eval()
        correspondences = model(images)
        return correspondences["confidence"].mean().cpu().numpy()

    def recon_metrics(self, images, key, trainer, pl_module):
        targets = denormalize(images["targets"]).cpu().numpy()
        output_images = denormalize(images[key.split("/")[-1]])
        self.psnr[key].append(self.calculate_psnr(targets, output_images.cpu().numpy()))
        self.ssim[key].append(self.calculate_ssim(targets, output_images.cpu().numpy()))
        self.mvsc[key].append(self.calculate_mvsc(pl_module.matcher, output_images))
        if len(self.mvsc[key]) >= self.cache_size:
            pl_module.logger.experiment.add_scalar(key + "_psnr", np.mean(self.psnr[key]), trainer.global_step)
            pl_module.logger.experiment.add_scalar(key + "_ssim", np.mean(self.ssim[key]), trainer.global_step)
            pl_module.logger.experiment.add_scalar(key + "_mvsc", np.mean(self.mvsc[key]), trainer.global_step)

    def recon_fid(self, images, key, trainer, pl_module):
        output_images = denormalize(images[key.split("/")[-1]])
        self.act2[key].append(self.calculate_activation_statistics(pl_module.evaluation, output_images))
        if len(self.act2[key]) >= self.cache_size:
            act1, act2 = np.concatenate(self.act1), np.concatenate(self.act2[key])
            mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
            mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
            pl_module.logger.experiment.add_scalar(
                key + "_fid",
                self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2),
                trainer.global_step,
            )

    def reset_cache(self):
        self.act1 = []
        self.act2 = dict()
        self.psnr = dict()
        self.ssim = dict()
        self.mvsc = dict()
        self.pd_bboxes_list = []
        self.gt_bboxes_list = []
        self.det_logged = False

    def has_recon_metrics(self, pl_module):
        return getattr(pl_module, "evaluation", None) is not None and getattr(pl_module, "matcher", None) is not None

    def should_process_batch(self, batch, pl_module):
        return self.has_recon_metrics(pl_module) or (self.eval_det_cfg is not None and "gt_bboxes" in batch)

    def recon_keys(self, prefix):
        keys = [
            f"{prefix}/recons",
            f"{prefix}/gen",
            f"{prefix}/cond_gen",
            f"{prefix}/ego_cond_gen",
            f"{prefix}/super_res_gen",
        ]
        for scale in range(10):
            keys.append(f"{prefix}/cond_gen_{scale}")
        for angle in range(0, 360, 30):
            keys.append(f"{prefix}/recons_rot_{angle}")
        for dx in [4, 2, 1, -1, -2, -4]:
            keys.append(f"{prefix}/recons_dx_{dx}")
        for dy in [4, 2, 1, -1, -2, -4]:
            keys.append(f"{prefix}/recons_dy_{dy}")
        return keys

    def process_recon_metrics(self, images, prefix, trainer, pl_module):
        if not self.has_recon_metrics(pl_module) or "targets" not in images:
            return

        targets = denormalize(images["targets"])
        self.act1.append(self.calculate_activation_statistics(pl_module.evaluation, targets))
        targets_key = f"{prefix}/targets"
        if targets_key not in self.mvsc:
            self.mvsc[targets_key] = []
        self.mvsc[targets_key].append(self.calculate_mvsc(pl_module.matcher, targets))
        if len(self.mvsc[targets_key]) >= self.cache_size:
            pl_module.logger.experiment.add_scalar(
                targets_key + "_mvsc",
                np.mean(self.mvsc[targets_key]),
                trainer.global_step,
            )

        for key in self.recon_keys(prefix):
            if key.split("/")[-1] in images:
                if key not in self.act2:
                    self.act2[key] = []
                    self.psnr[key] = []
                    self.ssim[key] = []
                    self.mvsc[key] = []
                self.recon_fid(images, key, trainer, pl_module)
                self.recon_metrics(images, key, trainer, pl_module)

    def gather_eval_tensor(self, tensor, trainer, pl_module):
        tensor = tensor.detach().cpu()
        if getattr(trainer, "world_size", 1) <= 1:
            return tensor
        gathered = pl_module.all_gather(tensor.to(pl_module.device))
        return gathered.detach().cpu().reshape(-1, *tensor.shape[1:])

    def process_det_metrics(self, images, prefix, trainer, pl_module):
        if self.eval_det_cfg is None or self.det_logged:
            return
        if "pd_bboxes" not in images or "gt_bboxes" not in images:
            return

        self.pd_bboxes_list.append(images["pd_bboxes"].detach().cpu())
        self.gt_bboxes_list.append(images["gt_bboxes"].detach().cpu())
        if len(self.pd_bboxes_list) < self.cache_size:
            return

        total_pd_bboxes = torch.cat(self.pd_bboxes_list, dim=0)
        total_gt_bboxes = torch.cat(self.gt_bboxes_list, dim=0)
        total_pd_bboxes = self.gather_eval_tensor(total_pd_bboxes, trainer, pl_module)
        total_gt_bboxes = self.gather_eval_tensor(total_gt_bboxes, trainer, pl_module)
        det_metrics = calc_det_metrics(total_pd_bboxes, total_gt_bboxes, self.eval_det_cfg)
        self.det_logged = True

        if getattr(trainer, "is_global_zero", True):
            for metric_name, metric_value in det_metrics.items():
                pl_module.logger.experiment.add_scalar(
                    f"{prefix}/det_{metric_name}",
                    metric_value,
                    trainer.global_step,
                )

    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset_cache()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.epoch_interval != 0:
            return
        if not self.should_process_batch(batch, pl_module):
            return

        images = pl_module.log_images(batch)
        self.process_recon_metrics(images, "val", trainer, pl_module)
        self.process_det_metrics(images, "val", trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        self.reset_cache()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.should_process_batch(batch, pl_module):
            return

        images = pl_module.log_images(batch)
        self.process_recon_metrics(images, "test", trainer, pl_module)
        self.process_det_metrics(images, "test", trainer, pl_module)
