from typing import Optional, Tuple, Dict, Union, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.checkpoint as checkpoint
from pytorch_lightning import LightningModule

from bev_vae.data.datasets.utils import bbox2cuboid, render_cuboids, plot_maps
from torchvision.transforms import functional as F
from bev_vae.models.perception.task_sharing import resample_bev_tokens


class Perceptor(LightningModule):
    def __init__(
        self,
        load_ema: bool,
        monitor: str,
        backbone: nn.Module,
        scene_encoder: nn.Module,
        connector: nn.Module,
        detector: nn.Module,
        segmentor: nn.Module,
        mapper: nn.Module,
        base_lr: float,
        grid_mask: Optional[nn.Module] = None,
        perspective_aux_head: Optional[nn.Module] = None,
        sparse4d_head: Optional[nn.Module] = None,
        weight_decay: float = 0.01,
        backbone_lr_mult: float = 0.1,
        backbone_adapter_lr_mult: float = 1.0,
        det_loss_weight: float = 1.0,
        seg_loss_weight: float = 1.0,
        map_loss_weight: float = 1.0,
        perspective_aux_loss_weight: float = 1.0,
        sparse4d_loss_weight: float = 1.0,
        scheduler_frequency: int = 1,
        scheduler = None,
        ckpt_path=None,
        ignore_keys=[],
        task_router: Optional[nn.Module] = None,
        map_feature_resampling: bool = False) -> None:
        super().__init__()
        
        self.load_ema = load_ema
        self.monitor = monitor
        self.backbone = backbone
        self.grid_mask = grid_mask
        self.scene_encoder = scene_encoder
        self.connector = connector
        self.detector = detector
        self.segmentor = segmentor
        self.mapper = mapper
        self.perspective_aux_head = perspective_aux_head
        self.sparse4d_head = sparse4d_head
        self.task_router = task_router
        self.map_feature_resampling = bool(map_feature_resampling)
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.backbone_lr_mult = backbone_lr_mult
        self.backbone_adapter_lr_mult = backbone_adapter_lr_mult
        self.det_loss_weight = det_loss_weight
        self.seg_loss_weight = seg_loss_weight
        self.map_loss_weight = map_loss_weight
        self.perspective_aux_loss_weight = perspective_aux_loss_weight
        self.sparse4d_loss_weight = sparse4d_loss_weight
        self.scheduler_frequency = scheduler_frequency
        self.scheduler = scheduler

        if self.task_router is not None:
            available = {"det": self.detector, "seg": self.segmentor, "map": self.mapper}
            weights = {"det": det_loss_weight, "seg": seg_loss_weight, "map": map_loss_weight}
            for task in self.task_router.tasks:
                if available[task] is None or weights[task] <= 0:
                    raise ValueError(f"Task router requires an active, positively weighted {task} BEV head; "
                                     "Sparse4D/perspective heads do not consume this BEV")
        if self.map_feature_resampling:
            if self.mapper is None or not all(hasattr(self.mapper, name) for name in ("pc", "scene_size")):
                raise ValueError("Map feature resampling requires a mapper with pc and scene_size")
            if not all(hasattr(self.scene_encoder, name) for name in ("pc", "scene_size")):
                raise ValueError("Map feature resampling requires scene_encoder pc and scene_size")

        self.params = (
            list(self.scene_encoder.parameters())+
            list(self.connector.parameters()))
        if self.detector is not None:
            self.params += list(self.detector.parameters()) 
        if self.segmentor is not None:
            self.params += list(self.segmentor.parameters()) 
        if self.mapper is not None:              
            self.params += list(self.mapper.parameters())
        if self.perspective_aux_head is not None:
            self.params += list(self.perspective_aux_head.parameters())
        if self.sparse4d_head is not None:
            self.params += list(self.sparse4d_head.parameters())
        if self.task_router is not None:
            self.params += list(self.task_router.parameters())
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

    def _add_optional_img_metas(self, batch: Tuple[Any, Any], img_metas: Dict) -> Dict:
        for key in ("gt_map_labels", "gt_map_points", "gt_map_orders", "gt_map_order_mask",
                    "gt_map_bev_seg", "gt_map_pv_seg"):
            if key in batch:
                img_metas[key] = batch[key]
        if 'aug_param' in batch:
            img_metas['aug_param'] = batch['aug_param']
        for key in ('timestamp_ns', 'ego_to_global', 'sequence_id', 'sequence_start', 'frame_index'):
            if key in batch:
                img_metas[key] = batch[key]
        if 'split' in batch:
            splits = batch['split']
            img_metas['temporal_scope'] = splits if isinstance(splits, str) else splits[0]
        return img_metas

    def _extract_bda_mat(self, img_metas: dict, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        aug_param = img_metas.get("aug_param")
        if aug_param is None:
            return None

        bda_mat = None
        only_gt = None
        if isinstance(aug_param, dict):
            if "GlobalRotScaleTransImage_param" in aug_param:
                param = aug_param["GlobalRotScaleTransImage_param"]
                bda_mat = param[4]
                only_gt = param[5] if len(param) > 5 else None
            elif "bda_mat" in aug_param:
                bda_mat = aug_param["bda_mat"]
                only_gt = aug_param.get("only_gt")
        if bda_mat is None:
            return None

        if only_gt is not None:
            only_gt = torch.as_tensor(only_gt, device=device)
            if not bool(torch.all(only_gt)):
                return None

        bda_mat = torch.as_tensor(bda_mat, device=device, dtype=dtype)
        if bda_mat.ndim == 2:
            bda_mat = bda_mat.unsqueeze(0)
        return bda_mat

    def apply_bev_coordinate_augmentation(self, scene_feats: torch.Tensor, img_metas: dict) -> torch.Tensor:
        bda_mat = self._extract_bda_mat(img_metas, scene_feats.device, scene_feats.dtype)
        if bda_mat is None:
            return scene_feats

        b, c, d, h, w = scene_feats.shape
        if bda_mat.shape[0] == 1 and b > 1:
            bda_mat = bda_mat.repeat(b, 1, 1)
        if bda_mat.shape[0] != b:
            raise ValueError(f"bda_mat batch size {bda_mat.shape[0]} does not match scene feature batch size {b}")

        pc = torch.as_tensor(self.scene_encoder.pc, device=scene_feats.device, dtype=scene_feats.dtype)
        x_centers = (torch.arange(w, device=scene_feats.device, dtype=scene_feats.dtype) + 0.5)
        x_centers = x_centers / w * (pc[3] - pc[0]) + pc[0]
        y_centers = (torch.arange(h, device=scene_feats.device, dtype=scene_feats.dtype) + 0.5)
        y_centers = y_centers / h * (pc[4] - pc[1]) + pc[1]
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")
        out_xy = torch.stack([xx, yy], dim=-1).reshape(1, h * w, 2).repeat(b, 1, 1)
        src_xy = torch.bmm(out_xy, bda_mat[:, :2, :2].transpose(1, 2)).reshape(b, h, w, 2)

        grid = torch.empty_like(src_xy)
        grid[..., 0] = (src_xy[..., 0] - pc[0]) / (pc[3] - pc[0]) * 2.0 - 1.0
        grid[..., 1] = (src_xy[..., 1] - pc[1]) / (pc[4] - pc[1]) * 2.0 - 1.0

        bev = scene_feats.reshape(b, c * d, h, w)
        bev = torch_F.grid_sample(bev, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return bev.reshape(b, c, d, h, w)

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        if self.load_ema and 'optimizer_states' in ckpt:
            print("Load EMA params for Perceptor.")
            ema_params = ckpt['optimizer_states'][0]['ema']
            for i in range((len(ema_params))):
                self.params[i].data.copy_(ema_params[i])
        print(f"Restored from {path}")
    
    def combine_all_views(self, image: torch.FloatTensor) -> torch.FloatTensor:
        return rearrange(image, 'b v ... -> (b v) ...')

    def extract_image_features(self, image: torch.FloatTensor):
        if self.grid_mask is not None:
            image = self.grid_mask(image)
        return self.backbone(image)
    
    def combine_all_frames(self, image: torch.FloatTensor) -> torch.FloatTensor:
        return rearrange(image, 'b f v ... -> (b f v) ...')

    def get_input(self, batch: Tuple[Any, Any], key: str) -> Any:
        x = batch[key] 
        if len(x.shape) == 3:
            return x.unsqueeze(1) # B1HW
        elif len(x.shape) == 5:
            return self.combine_all_views(x) # (BV)CHW
        elif len(x.shape) == 6:
            return self.combine_all_frames(x) # (BSV)CHW
        else:
            return x # BCHW

    def get_task_features(self, z: torch.Tensor, task: str) -> torch.Tensor:
        """Shared training/export/inference path; never consumes labels.

        Routing happens on the full detection BEV. Map resampling then projects
        cell centres into its own physical region; map gradients are zero
        outside the crop's interpolation support at this connector boundary.
        """
        features = self.task_router(z, task) if (
            self.task_router is not None and task in self.task_router.tasks
        ) else z
        if task == "map" and self.map_feature_resampling:
            features = resample_bev_tokens(features, self.scene_encoder.pc,
                                          self.scene_encoder.scene_size,
                                          self.mapper.pc, self.mapper.scene_size)
        return features
    
    def forward(self, image: torch.FloatTensor, img_metas: dict, split: str) -> torch.FloatTensor:
        img_feats = self.extract_image_features(image)
        needs_scene_feats = any(
            head is not None for head in (self.detector, self.segmentor, self.mapper)
        )
        needs_perspective_feats = (
            self.perspective_aux_head is not None or self.sparse4d_head is not None
            or getattr(self.mapper, "requires_perspective_features", False)
        )
        if needs_scene_feats:
            if needs_perspective_feats:
                scene_feats, perspective_feats = self.scene_encoder(
                    img_feats, img_metas, return_img_feats=True
                )
            else:
                scene_feats = self.scene_encoder(img_feats, img_metas)
                perspective_feats = None
            scene_feats = self.apply_bev_coordinate_augmentation(scene_feats, img_metas)
            z = self.connector(rearrange(scene_feats, 'b c d h w -> b (h w) (d c)'))
        else:
            scene_feats = None
            z = None
            perspective_feats = (
                self.scene_encoder.encode_image_features(img_feats, img_metas)
                if needs_perspective_feats
                else None
            )
        losses, log_dict = [], {}
        task_losses = {}
        proposals = None
        if self.perspective_aux_head is not None:
            use_proposals = self.detector is not None and getattr(
                self.detector, "proposal_query_count", 0
            ) > 0
            if use_proposals:
                aux_loss, aux_log_dict, proposals = self.perspective_aux_head(
                    perspective_feats,
                    img_metas,
                    split,
                    return_proposals=True,
                )
            else:
                aux_loss, aux_log_dict = self.perspective_aux_head(
                    perspective_feats,
                    img_metas,
                    split,
                )
            losses.append(self.perspective_aux_loss_weight * aux_loss)
            log_dict.update({f"perspective_aux/{k}": v for k, v in aux_log_dict.items()})
            log_dict[f"perspective_aux/{split}/weighted_loss"] = (
                self.perspective_aux_loss_weight * aux_loss
            ).item()
        if self.detector is not None:
            det_loss, det_log_dict = self.detector(self.get_task_features(z, "det"), img_metas,
                                                  split, proposals=proposals)
            losses.append(self.det_loss_weight * det_loss)
            task_losses["det"] = self.det_loss_weight * det_loss
            log_dict.update({f"det/{k}": v for k, v in det_log_dict.items()})
            log_dict[f"det/{split}/weighted_loss"] = (self.det_loss_weight * det_loss).item()
        if self.sparse4d_head is not None:
            sparse4d_loss, sparse4d_log_dict = self.sparse4d_head(
                perspective_feats, img_metas, split
            )
            losses.append(self.sparse4d_loss_weight * sparse4d_loss)
            log_dict.update(
                {f"sparse4d/{k}": v for k, v in sparse4d_log_dict.items()}
            )
            log_dict[f"sparse4d/{split}/weighted_loss"] = (
                self.sparse4d_loss_weight * sparse4d_loss
            ).item()
        if self.segmentor is not None:
            seg_loss, seg_log_dict = self.segmentor(self.get_task_features(z, "seg"), img_metas, split)
            losses.append(self.seg_loss_weight * seg_loss)
            task_losses["seg"] = self.seg_loss_weight * seg_loss
            log_dict.update({f"seg/{k}": v for k, v in seg_log_dict.items()})
            log_dict[f"seg/{split}/weighted_loss"] = (self.seg_loss_weight * seg_loss).item()
        if self.mapper is not None:
            kwargs = {"perspective_feats": perspective_feats} if getattr(
                self.mapper, "requires_perspective_features", False) else {}
            map_loss, map_log_dict = self.mapper(self.get_task_features(z, "map"), img_metas, split, **kwargs)
            losses.append(self.map_loss_weight * map_loss)
            task_losses["map"] = self.map_loss_weight * map_loss
            log_dict.update({f"map/{k}": v for k, v in map_log_dict.items()})
            log_dict[f"map/{split}/weighted_loss"] = (self.map_loss_weight * map_loss).item()
        if (self.task_router is not None and self.training and split == "train"
                and torch.is_grad_enabled() and self.task_router.should_measure(self.global_step)):
            routing_loss, routing_logs = self.task_router.compatibility_loss(
                task_losses, z, step=self.global_step)
            losses.append(routing_loss)
            log_dict.update({f"sharing/{split}/{key}": value for key, value in routing_logs.items()})
        return sum(losses), log_dict

    def on_train_epoch_start(self):
        loader = self.trainer.train_dataloader
        sampler = getattr(loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        if self.sparse4d_head is not None:
            self.sparse4d_head.reset_temporal("train")

    def on_validation_epoch_start(self):
        if self.sparse4d_head is not None:
            self.sparse4d_head.reset_temporal("val")

    def on_test_epoch_start(self):
        if self.sparse4d_head is not None:
            self.sparse4d_head.reset_temporal("test")

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
        x = self.get_input(batch, "imgs")
        img_metas = {k: batch[k] for k in ['geometric', 'intrinsic', 'extrinsic']}
        img_metas = self._add_optional_img_metas(batch, img_metas)
        img_metas['cam_num'] = len(batch['cam_names'])
        img_metas['img_shape'] = x.shape[-2:]
        if 'gt_bboxes' in batch:
            img_metas['gt_bboxes'] = batch['gt_bboxes']
        if 'gt_map' in batch:
            img_metas['gt_map'] = batch['gt_map']
        if 'gt_map_vectors' in batch:
            img_metas['gt_map_vectors'] = batch['gt_map_vectors']
        loss, log_dict = self(x[:, -3:], img_metas, split="train")
        self.log("train/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log_dict(log_dict, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
        x = self.get_input(batch, "imgs")
        img_metas = {k: batch[k] for k in ['geometric', 'intrinsic', 'extrinsic']}
        img_metas = self._add_optional_img_metas(batch, img_metas)
        img_metas['cam_num'] = len(batch['cam_names'])
        img_metas['img_shape'] = x.shape[-2:]
        if 'gt_bboxes' in batch:
            img_metas['gt_bboxes'] = batch['gt_bboxes']
        if 'gt_map' in batch:
            img_metas['gt_map'] = batch['gt_map']
        if 'gt_map_vectors' in batch:
            img_metas['gt_map_vectors'] = batch['gt_map_vectors']
        loss, log_dict = self(x[:, -3:], img_metas, split="val")
        self.log("val/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log_dict(log_dict, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        return loss

    def test_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
        x = self.get_input(batch, "imgs")
        img_metas = {k: batch[k] for k in ['geometric', 'intrinsic', 'extrinsic']}
        img_metas = self._add_optional_img_metas(batch, img_metas)
        img_metas['cam_num'] = len(batch['cam_names'])
        img_metas['img_shape'] = x.shape[-2:]
        if 'gt_bboxes' in batch:
            img_metas['gt_bboxes'] = batch['gt_bboxes']
        if 'gt_map' in batch:
            img_metas['gt_map'] = batch['gt_map']
        if 'gt_map_vectors' in batch:
            img_metas['gt_map_vectors'] = batch['gt_map_vectors']
        loss, log_dict = self(x[:, -3:], img_metas, split="test")
        self.log("test/total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log_dict(log_dict, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        return loss

    def on_before_optimizer_step(self, optimizer):
        if self.sparse4d_head is None:
            return
        interval = max(1, self.trainer.log_every_n_steps)
        if self.global_step % interval != 0:
            return
        # Lightning calls this after AMP unscaling and before gradient clipping.
        # Its global_step can advance even when GradScaler skips AdamW.step().
        gradients = [
            parameter.grad.detach()
            for parameter in self.sparse4d_head.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return
        probe = self.sparse4d_head.layers[0].refine.cls_layers[-1].weight
        adam_step = optimizer.state.get(probe, {}).get("step", 0)
        if isinstance(adam_step, torch.Tensor):
            adam_step = adam_step.detach().item()
        values = {
            "sparse4d/train/nonfinite_grad_tensors": torch.stack([
                (~torch.isfinite(grad).all()).float() for grad in gradients
            ]).sum(),
            "sparse4d/train/grad_norm_before_clip": torch.stack([
                grad.float().norm() for grad in gradients
            ]).norm(),
            "sparse4d/train/adam_step_before_update": float(adam_step),
        }
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            values["sparse4d/train/amp_scale"] = scaler.get_scale()
        self.log_dict(values, on_step=True, on_epoch=False, sync_dist=True)

    def configure_optimizers(self):
        adapter_parameter_fn = getattr(self.backbone, "adapter_parameters", None)
        if callable(adapter_parameter_fn):
            adapter_params = [param for param in adapter_parameter_fn() if param.requires_grad]
            adapter_ids = {id(param) for param in adapter_params}
            pretrained_backbone_params = [
                param
                for param in self.backbone.parameters()
                if param.requires_grad and id(param) not in adapter_ids
            ]
        else:
            adapter_params = []
            pretrained_backbone_params = [
                param for param in self.backbone.parameters() if param.requires_grad
            ]

        param_groups = []
        if pretrained_backbone_params:
            param_groups.append(
                {
                    "params": pretrained_backbone_params,
                    "lr": self.base_lr * self.backbone_lr_mult,
                }
            )
        if adapter_params:
            param_groups.append(
                {
                    "params": adapter_params,
                    "lr": self.base_lr * self.backbone_adapter_lr_mult,
                }
            )
        param_groups.append({"params": self.params, "lr": self.base_lr})
        optimizer = torch.optim.AdamW(param_groups, lr=self.base_lr, weight_decay=self.weight_decay)
        scheduler = [{
            'scheduler': LambdaLR(optimizer, lr_lambda=self.scheduler.schedule),
            'interval': 'step',
            'frequency': self.scheduler_frequency}] if self.scheduler is not None else []
        return [optimizer], scheduler

    @torch.no_grad()
    def log_images(self, batch: Tuple[Any, Any]) -> Dict:
        img_metas = {k: batch[k] for k in ['geometric', 'intrinsic', 'extrinsic']}
        img_metas = self._add_optional_img_metas(batch, img_metas)
        img_metas['cam_num'] = len(batch['cam_names'])
        if 'gt_bboxes' in batch:
            img_metas['gt_bboxes'] = batch['gt_bboxes']
        if 'gt_map_vectors' in batch:
            img_metas['gt_map_vectors'] = batch['gt_map_vectors']

        log = dict()
        img_metas["temporal_read_only"] = True

        x = self.get_input(batch, "imgs")
        img_metas['img_shape'] = x.shape[-2:]
        
        inputs = x[:, -3:]
        log["inputs"] = inputs.clone()
        log["targets"] = x[:, :3].clone()
 
        img_feats = self.extract_image_features(inputs)
        proposals = None
        needs_scene_feats = any(
            head is not None for head in (self.detector, self.segmentor, self.mapper)
        )
        needs_perspective_feats = (
            self.perspective_aux_head is not None or self.sparse4d_head is not None
            or getattr(self.mapper, "requires_perspective_features", False)
        )
        if needs_scene_feats:
            if needs_perspective_feats:
                scene_feats, perspective_feats = self.scene_encoder(
                    img_feats, img_metas, return_img_feats=True
                )
            else:
                scene_feats = self.scene_encoder(img_feats, img_metas)
                perspective_feats = None
            scene_feats = self.apply_bev_coordinate_augmentation(scene_feats, img_metas)
            z = self.connector(rearrange(scene_feats, 'b c d h w -> b (h w) (d c)'))
        else:
            scene_feats = None
            z = None
            perspective_feats = (
                self.scene_encoder.encode_image_features(img_feats, img_metas)
                if needs_perspective_feats
                else None
            )
        if self.perspective_aux_head is not None and getattr(
            self.detector, "proposal_query_count", 0
        ) > 0:
            proposals = self.perspective_aux_head.get_proposals(
                perspective_feats, img_metas
            )
        pd_bboxes = None
        if self.detector is not None:
            pd_bboxes = self.detector.get_bboxes(
                self.get_task_features(z, "det"), score_thr=0.4, proposals=proposals
            ) # max_num=300
            log["detector_pd_bboxes"] = pd_bboxes

        if self.sparse4d_head is not None:
            sparse4d_pd_bboxes = self.sparse4d_head.get_bboxes(
                perspective_feats,
                img_metas,
                score_thr=0.4,
                max_num=300,
            )
            log["sparse4d_pd_bboxes"] = sparse4d_pd_bboxes
            if pd_bboxes is None:
                pd_bboxes = sparse4d_pd_bboxes

        if self.mapper is not None:
            pd_map_vectors = self.mapper.get_map_vectors(
                self.get_task_features(z, "map"), score_thr=0.4)  # max_num=50
        
        if 'gt_bboxes' in batch and pd_bboxes is not None:
            pil_imgs_gt = render_cuboids(bbox2cuboid(batch['gt_bboxes']), x[:, -3:], img_metas)
            # pil_imgs_pd = render_cuboids(bbox2cuboid(pd_bboxes), dec, img_metas)
            pil_imgs_pd = render_cuboids(bbox2cuboid(pd_bboxes), inputs, img_metas)
            imgs_gt_bboxes = torch.stack([F.to_tensor(img) for img in pil_imgs_gt])
            imgs_pd_bboxes = torch.stack([F.to_tensor(img) for img in pil_imgs_pd])
            log["pd_bboxes"] = pd_bboxes
            log["gt_bboxes"] = batch['gt_bboxes']
            log["imgs_gt_bboxes"] = imgs_gt_bboxes
            log["imgs_pd_bboxes"] = imgs_pd_bboxes
        if 'gt_map_vectors' in batch:
            pil_maps_gt = plot_maps(batch['gt_map_vectors'], self.mapper.vertex_num, self.mapper.pc)
            log["gt_map_vectors"] = torch.stack([F.to_tensor(img.convert("RGB")) for img in pil_maps_gt])
            pil_maps_pd = plot_maps(pd_map_vectors, self.mapper.vertex_num)
            log["pd_map_vectors"] = torch.stack([F.to_tensor(img.convert("RGB")) for img in pil_maps_pd])
        return log
