"""Local shared/private BEV adapters with an optional first-order teaching signal.

The cosine of task gradients is a diagnostic proxy for compatibility, not a
causal estimate of transfer. Labels and task gradients are never inference
inputs. There is no gradient surgery or second-order differentiation here.
"""
from typing import Dict, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def gradient_compatibility(first, second, norm_eps=1e-8):
    """Detached float32 cosine targets at each [B, HW] position.

    A token is supervised only when BOTH task gradients are finite and have
    nonzero norm. This is numerical co-gradient support, not a visibility or
    annotation mask. A map crop includes its bilinear interpolation halo.
    Nonfinite entries are masked before arithmetic, avoiding NaN * 0 losses.
    """
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("Task gradients must have matching [B, HW, C] shapes")
    first, second = first.detach().float(), second.detach().float()
    finite = torch.isfinite(first).all(-1) & torch.isfinite(second).all(-1)
    first = torch.where(torch.isfinite(first), first, torch.zeros_like(first))
    second = torch.where(torch.isfinite(second), second, torch.zeros_like(second))
    first_norm, second_norm = first.norm(dim=-1), second.norm(dim=-1)
    valid = finite & torch.isfinite(first_norm) & torch.isfinite(second_norm)
    valid = valid & (first_norm > norm_eps) & (second_norm > norm_eps)
    first_unit = first / first_norm.clamp_min(norm_eps).unsqueeze(-1)
    second_unit = second / second_norm.clamp_min(norm_eps).unsqueeze(-1)
    cosine = (first_unit * second_unit).sum(-1).clamp(-1, 1)
    cosine = torch.where(valid, cosine, torch.zeros_like(cosine))
    return {
        "target": (cosine + 1) * 0.5,
        "cosine": cosine,
        "valid": valid,
        "finite": finite & torch.isfinite(first_norm) & torch.isfinite(second_norm),
        "first_norm": torch.where(valid, first_norm, torch.zeros_like(first_norm)),
        "second_norm": torch.where(valid, second_norm, torch.zeros_like(second_norm)),
    }


def resample_bev_tokens(z, source_pc, source_shape, target_pc, target_shape):
    """Resample cell centres in physical coordinates, preserving x/y ordering."""
    h, w = (int(v) for v in source_shape[-2:])
    th, tw = (int(v) for v in target_shape[-2:])
    if z.ndim != 3 or z.shape[1] != h * w or min(h, w, th, tw) <= 0:
        raise ValueError("BEV token count does not match the source grid")
    if any(source_pc[i + 3] <= source_pc[i] or target_pc[i + 3] <= target_pc[i]
           for i in (0, 1)):
        raise ValueError("BEV x/y ranges must have positive extent")
    if any(target_pc[i] < source_pc[i] - 1e-6 or
           target_pc[i + 3] > source_pc[i + 3] + 1e-6 for i in (0, 1)):
        raise ValueError("The map coordinate range must be inside the shared BEV")
    if (h, w) == (th, tw) and all(source_pc[i] == target_pc[i] for i in (0, 1, 3, 4)):
        return z
    # Explicit float32 sampling also works with CPU bfloat16 and AMP inputs.
    with torch.autocast(device_type=z.device.type, enabled=False):
        xs = (torch.arange(tw, device=z.device, dtype=torch.float32) + 0.5) / tw
        ys = (torch.arange(th, device=z.device, dtype=torch.float32) + 0.5) / th
        xs = xs * (target_pc[3] - target_pc[0]) + target_pc[0]
        ys = ys * (target_pc[4] - target_pc[1]) + target_pc[1]
        ys, xs = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((2 * (xs - source_pc[0]) / (source_pc[3] - source_pc[0]) - 1,
                            2 * (ys - source_pc[1]) / (source_pc[4] - source_pc[1]) - 1), -1)
        grid = grid.unsqueeze(0).expand(z.shape[0], -1, -1, -1)
        bev = z.float().transpose(1, 2).reshape(z.shape[0], z.shape[-1], h, w)
        sampled = F.grid_sample(bev, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return sampled.flatten(2).transpose(1, 2).to(z.dtype)


class ResidualAdapter(nn.Sequential):
    def __init__(self, embed_dim, adapter_dim):
        super().__init__(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, adapter_dim),
                         nn.GELU(), nn.Linear(adapter_dim, embed_dim))
        # Every experiment starts from the exact shared features.
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


class TaskSharingRouter(nn.Module):
    """Two-task, token-local mixture of shared and task-private residuals.

    ``shared`` is identity; ``private`` adds an independent adapter; ``uniform``
    averages shared/private adapters. ``learned`` and ``compatibility`` have
    identical architectures; only the latter supervises gates. All losses are
    accumulated into the normal single backward call by the training wrapper.
    """
    MODES = {"shared", "private", "uniform", "learned", "compatibility"}

    def __init__(self, embed_dim=256, adapter_dim=64, gate_hidden_dim=32,
                 tasks: Sequence[str] = ("det", "map"), mode="compatibility",
                 target_mode="local", residual_scale=0.1, supervision_weight=0.05,
                 diagnostic_interval=16, warmup_steps=500, norm_eps=1e-8,
                 fp16_probe_scale=1024.0):
        super().__init__()
        self.tasks = tuple(tasks)
        if len(self.tasks) != 2 or len(set(self.tasks)) != 2:
            raise ValueError("This first-order implementation requires exactly two distinct tasks")
        if not set(self.tasks).issubset({"det", "seg", "map"}):
            raise ValueError("Only heads consuming the shared BEV can be routed")
        if mode not in self.MODES or target_mode not in {"local", "global", "shuffled"}:
            raise ValueError("Unknown task-sharing mode or compatibility target mode")
        if min(embed_dim, adapter_dim, gate_hidden_dim, diagnostic_interval) <= 0:
            raise ValueError("Dimensions and diagnostic_interval must be positive")
        if (warmup_steps < 0 or supervision_weight < 0 or residual_scale <= 0
                or norm_eps <= 0 or fp16_probe_scale < 1):
            raise ValueError("Invalid task-sharing loss/scale/schedule settings")
        self.embed_dim, self.mode, self.target_mode = embed_dim, mode, target_mode
        self.residual_scale, self.supervision_weight = residual_scale, supervision_weight
        self.diagnostic_interval, self.warmup_steps = int(diagnostic_interval), int(warmup_steps)
        self.norm_eps = norm_eps
        self.fp16_probe_scale = float(fp16_probe_scale)
        self.private = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        self.shared = None
        if mode != "shared":
            self.private = nn.ModuleDict({t: ResidualAdapter(embed_dim, adapter_dim) for t in self.tasks})
        if mode in {"uniform", "learned", "compatibility"}:
            self.shared = ResidualAdapter(embed_dim, adapter_dim)
        if mode in {"learned", "compatibility"}:
            for task in self.tasks:
                gate = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, gate_hidden_dim),
                                     nn.GELU(), nn.Linear(gate_hidden_dim, 1))
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)
                self.gates[task] = gate

    def forward(self, z, task):
        if task not in self.tasks:
            raise ValueError(f"Task {task!r} is not configured in {self.tasks}")
        if z.ndim != 3 or z.shape[-1] != self.embed_dim:
            raise ValueError(f"Router expects [B, HW, {self.embed_dim}], got {tuple(z.shape)}")
        if self.mode == "shared":
            return z
        private = self.private[task](z)
        if self.mode == "private":
            update = private
        else:
            share = 0.5 if self.mode == "uniform" else self.gates[task](z).sigmoid()
            update = share * self.shared(z) + (1 - share) * private
        return z + self.residual_scale * update

    def should_measure(self, step):
        # global_step is identical on all ranks, including gradient accumulation.
        return int(step) % self.diagnostic_interval == 0

    def _control_target(self, values, step):
        target, valid = values["target"], values["valid"]
        if self.target_mode == "global":
            # Per-example spatially constant target, averaged over joint support.
            count = valid.sum(-1, keepdim=True).clamp_min(1)
            return ((target * valid).sum(-1, keepdim=True) / count).expand_as(target)
        if self.target_mode == "shuffled":
            target = target.clone()
            generator = torch.Generator(device=target.device)
            generator.manual_seed(17011 + int(step))
            for row in range(target.shape[0]):
                indices = valid[row].nonzero(as_tuple=True)[0]
                permutation = torch.randperm(len(indices), generator=generator, device=target.device)
                target[row, indices] = target[row, indices[permutation]]
        return target

    def compatibility_loss(self, task_losses: Dict[str, torch.Tensor], z, step=0):
        """Measure unscaled weighted task gradients without accumulating .grad.

        Call only on scheduled training forwards with autograd enabled. The
        teacher and gate input for this auxiliary loss are detached. The
        ordinary task losses still differentiate through both gates and BEV.
        Distributed count normalization gives a true global valid-token mean
        after DDP averages parameter gradients, even for empty-support ranks.
        """
        if not torch.is_grad_enabled() or not z.requires_grad:
            raise RuntimeError("Compatibility diagnostics require differentiable training BEV features")
        if any(t not in task_losses for t in self.tasks):
            raise ValueError("Both routed tasks must supply their weighted training loss")
        gradients = []
        # z itself may be FP32 while a downstream AMP head contains FP16
        # operations, so checking z.dtype would miss interior underflow.
        probe_scale = self.fp16_probe_scale
        for task in self.tasks:
            loss = task_losses[task]
            # Probe before Lightning's scaled backward: compensate FP16 range,
            # then unscale in float32 before cosine/norm computation. This uses
            # no GradScaler state and never accumulates parameter gradients.
            probe_loss = loss.float() * probe_scale
            grad = torch.autograd.grad(probe_loss, z, retain_graph=True, create_graph=False,
                                       allow_unused=True)[0] if loss.requires_grad else None
            gradients.append(torch.zeros_like(z, dtype=torch.float32) if grad is None
                             else grad.detach().float() / probe_scale)
        with torch.autocast(device_type=z.device.type, enabled=False):
            values = gradient_compatibility(*gradients, norm_eps=self.norm_eps)
            valid = values["valid"]
            count = valid.sum().float()
            # Reduce sufficient statistics, not rank-local ratios. A rank with
            # no support must not dilute another rank's conflict measurement.
            statistics = torch.stack((count, count.new_tensor(valid.numel()),
                ((values["cosine"] < 0) & valid).sum().float(),
                (values["cosine"] * valid).sum(), (values["target"] * valid).sum(),
                values["first_norm"].sum(), values["second_norm"].sum(),
                (~values["finite"]).sum().float()))
            world_size = 1
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            if distributed:
                torch.distributed.all_reduce(statistics)
                world_size = torch.distributed.get_world_size()
            global_count, token_count = statistics[:2]
            denom = global_count.clamp_min(1)
            logs = {
                "valid_count": global_count,
                "valid_fraction": global_count / token_count.clamp_min(1),
                "conflict_fraction": statistics[2] / denom,
                "cosine": statistics[3] / denom,
                "target_mean": statistics[4] / denom,
                f"{self.tasks[0]}_grad_norm": statistics[5] / denom,
                f"{self.tasks[1]}_grad_norm": statistics[6] / denom,
                "nonfinite_fraction": statistics[7] / token_count.clamp_min(1),
            }
            # Graph-connected zero permits backward on an entirely empty mask.
            auxiliary = z.float().sum() * 0
            if self.gates:
                logits = [self.gates[t](z.detach().float()).squeeze(-1) for t in self.tasks]
                gate_probabilities = [value.detach().sigmoid() for value in logits]
                gate_statistics = torch.stack([torch.stack((value.sum(), (value * valid).sum(),
                    (value.square() * valid).sum())) for value in gate_probabilities])
                if distributed:
                    torch.distributed.all_reduce(gate_statistics)
                for task, statistics_for_task in zip(self.tasks, gate_statistics):
                    support_mean = statistics_for_task[1] / denom
                    support_variance = (statistics_for_task[2] / denom - support_mean.square()).clamp_min(0)
                    logs[f"{task}_share_gate"] = statistics_for_task[0] / token_count.clamp_min(1)
                    logs[f"{task}_support_gate_mean"] = support_mean
                    logs[f"{task}_support_gate_std"] = support_variance.sqrt()
                if self.mode == "compatibility" and int(step) >= self.warmup_steps:
                    target = self._control_target(values, step).detach()
                    numerator = sum((F.binary_cross_entropy_with_logits(value, target,
                                    reduction="none") * valid).sum() for value in logits)
                    auxiliary = (numerator * world_size / global_count.clamp_min(1)
                                 / len(self.tasks) * self.supervision_weight)
            logs["aux_loss"] = auxiliary.detach()
        return auxiliary, {name: value.detach() for name, value in logs.items()}
