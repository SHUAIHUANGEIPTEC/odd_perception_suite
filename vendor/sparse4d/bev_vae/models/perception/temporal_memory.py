"""Detached, per-batch-slot Sparse4D memory in BEV-VAE ego coordinates.

No tensors here are model parameters/checkpoint state. Each distributed process
owns its memory; train/validation/inference use separate namespaces.
"""
import torch


def gather_instances(values, indices):
    return values.gather(1, indices[..., None].expand(-1, -1, values.shape[-1]))


def project_temporal_anchors(anchor, previous_to_current, seconds):
    """Move by velocity in the previous ego frame, then change coordinates.

State: xyz, log(lwh), sin(yaw), cos(yaw), vx, vy, vz.
Transforms may include the independently sampled 3D training rotations.
"""
    with torch.autocast(device_type=anchor.device.type, enabled=False):
        anchor = anchor.float()
        transform = previous_to_current.float()
        rotation = transform[:, :3, :3]
        center = anchor[..., :3] + anchor[..., 8:11] * seconds.float()[:, None, None]
        center = torch.einsum('bij,bnj->bni', rotation, center) + transform[:, None, :3, 3]
        velocity = torch.einsum('bij,bnj->bni', rotation, anchor[..., 8:11])
        heading = torch.stack((anchor[..., 7], anchor[..., 6], torch.zeros_like(anchor[..., 6])), -1)
        heading = torch.einsum('bij,bnj->bni', rotation, heading)[..., :2]
        norm = heading.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(heading)
        fallback[..., 0] = 1
        heading = torch.where(norm > 1e-6, heading / norm.clamp_min(1e-6), fallback)
        return torch.cat((center, anchor[..., 3:6], heading[..., [1, 0]], velocity), -1)


class TemporalMemory:
    REQUIRED = ('timestamp_ns', 'ego_to_global', 'sequence_id', 'sequence_start', 'frame_index')

    def __init__(self, num_instances, confidence_decay=0.6, max_time_interval=2.0, default_time_interval=0.5,
                 min_time_interval=0.01):
        self.num_instances = int(num_instances)
        self.confidence_decay = float(confidence_decay)
        self.max_time_interval = float(max_time_interval)
        self.default_time_interval = float(default_time_interval)
        self.min_time_interval = float(min_time_interval)
        self.reset()

    def reset(self, scope=None):
        if scope is None:
            self.states = {}
            self.replays = {}
            self.next_ids = {}
        else:
            self.states.pop(scope, None)
            self.replays.pop(scope, None)
            self.next_ids.pop(scope, None)

    def prepare(self, metas, base_feature, base_anchor):
        missing = [key for key in self.REQUIRED if key not in metas]
        if missing:
            raise ValueError('Temporal Sparse4D requires sequence metadata: ' + ', '.join(missing))
        scope = metas.get('temporal_scope', 'inference')
        device = base_anchor.device
        batch_size = base_anchor.shape[0]
        timestamp = torch.as_tensor(metas['timestamp_ns'], device=device, dtype=torch.int64).reshape(-1)
        pose = torch.as_tensor(metas['ego_to_global'], device=device, dtype=torch.float64)
        frame_index = torch.as_tensor(metas['frame_index'], device=device, dtype=torch.int64).reshape(-1)
        starts = torch.as_tensor(metas['sequence_start'], device=device, dtype=torch.bool).reshape(-1)
        sequence_ids = tuple(metas['sequence_id'])
        if (timestamp.shape != (batch_size,) or frame_index.shape != (batch_size,)
                or starts.shape != (batch_size,) or pose.shape != (batch_size, 4, 4)
                or len(sequence_ids) != batch_size):
            raise ValueError('Temporal metadata batch dimensions do not match image features')
        if not torch.isfinite(pose).all():
            raise ValueError('Non-finite temporal ego pose')
        # Nanosecond integers avoid loss of sub-second precision at Unix epochs.
        key = (sequence_ids, tuple(timestamp.tolist()), tuple(frame_index.tolist()))
        read_only = bool(metas.get('temporal_read_only', False))
        replay = self.replays.get(scope)
        if read_only and replay is not None and replay['key'] == key:
            return replay

        k = self.num_instances
        context = dict(scope=scope, key=key, sequence_ids=sequence_ids,
                       timestamp=timestamp.detach().clone(), pose=pose.detach().clone(),
                       frame_index=frame_index.detach().clone(),
                       feature=base_feature[:, :k].detach().clone(),
                       anchor=base_anchor[:, :k].detach().clone(),
                       confidence=base_anchor.new_zeros(batch_size, k),
                       instance_ids=torch.full((batch_size, k), -1, dtype=torch.long, device=device),
                       valid=torch.zeros(batch_size, dtype=torch.bool, device=device),
                       interval=base_anchor.new_full((batch_size,), self.default_time_interval))
        previous = self.states.get(scope)
        if previous is not None and previous['anchor'].shape[0] == batch_size:
            same_sequence = torch.tensor([a == b for a, b in zip(sequence_ids, previous['sequence_ids'])], device=device)
            dt = (timestamp - previous['timestamp']).double() * 1e-9
            consecutive = same_sequence & ~starts & (frame_index == previous['frame_index'] + 1)
            too_short = consecutive & (dt > 0) & (dt < self.min_time_interval)
            if too_short.any():
                raise ValueError(
                    f"Temporal interval {dt[too_short].tolist()} seconds is too short; "
                    "check temporal_timestamp_unit: timestamp_ns must contain nanoseconds, "
                    "and nuScenes keyframes should be about 0.5 seconds apart"
                )
            valid = (consecutive
                     & (dt > 0) & (dt <= self.max_time_interval))
            # Compute transforms in FP64 before casting small relative poses.
            transform = torch.linalg.solve(pose, previous['pose'])
            # Invalid slots use a finite, bounded time interval too.
            interval = torch.where(valid, dt, dt.new_full(dt.shape, self.default_time_interval)).float()
            projected = project_temporal_anchors(previous['anchor'], transform, interval)
            # Older PyTorch versions only accept one dimension in Tensor.all.
            valid &= torch.isfinite(projected).flatten(start_dim=1).all(dim=1)
            context.update(
                valid=valid, interval=torch.where(valid, interval, context['interval']),
                feature=torch.where(valid[:, None, None], previous['feature'], context['feature']),
                anchor=torch.where(valid[:, None, None], projected, context['anchor']),
                confidence=torch.where(valid[:, None], previous['confidence'], context['confidence']))
            if 'instance_ids' in previous:
                context['instance_ids'] = torch.where(valid[:, None], previous['instance_ids'], context['instance_ids'])
        return context

    @torch.no_grad()
    def assign_instance_ids(self, classification, context, threshold=0.2, read_only=False):
        """Sparse4Dv3 query identity propagation; no post-hoc box association.

        IDs use int64, share exactly the feature/anchor cache ranking, and only
        survive consecutive valid frames. Read-only visualization cannot consume IDs.
        """
        confidence = classification.detach().sigmoid().amax(-1)
        ids = torch.full_like(confidence, -1, dtype=torch.long)
        ids[:, :self.num_instances] = context['instance_ids']
        new = (ids < 0) & (confidence >= threshold)
        start = self.next_ids.get(context['scope'], 0)
        count = int(new.sum())
        ids[new] = torch.arange(start, start + count, device=ids.device)
        if not read_only:
            self.next_ids[context['scope']] = start + count
        return ids

    def update(self, feature, anchor, classification, context, num_anchor):
        """After the first decoder, reserve K slots for history and N-K for new objects.

        DN queries remain at the end, outside ranking and memory.
        """
        k = self.num_instances
        indices = classification[:, :num_anchor].amax(-1).topk(num_anchor - k, dim=1).indices
        selected_feature = torch.cat((context['feature'], gather_instances(feature[:, :num_anchor], indices)), 1)
        selected_anchor = torch.cat((context['anchor'], gather_instances(anchor[:, :num_anchor], indices)), 1)
        valid = context['valid'][:, None, None]
        return (torch.cat((torch.where(valid, selected_feature, feature[:, :num_anchor]), feature[:, num_anchor:]), 1),
                torch.cat((torch.where(valid, selected_anchor, anchor[:, :num_anchor]), anchor[:, num_anchor:]), 1))

    @torch.no_grad()
    def cache(self, feature, anchor, classification, context, instance_ids=None):
        confidence = classification.detach().amax(-1).sigmoid()
        finite = (torch.isfinite(anchor).all(-1) & torch.isfinite(feature).all(-1)
                  & torch.isfinite(classification).all(-1))
        if not finite.all():
            raise FloatingPointError(
                'Non-finite Sparse4D output; refusing to poison temporal memory. '
                f'Bad slot/query indices: {(~finite).nonzero()[:8].tolist()}; '
                f'nonfinite anchor/feature/logits: '
                f'{int((~torch.isfinite(anchor)).sum())}/'
                f'{int((~torch.isfinite(feature)).sum())}/'
                f'{int((~torch.isfinite(classification)).sum())}; '
                f'interval_s={context["interval"].tolist()}, '
                f'sequences={context["sequence_ids"]}'
            )
        k = self.num_instances
        confidence[:, :k] = torch.maximum(confidence[:, :k], context['confidence'] * self.confidence_decay)
        scores, indices = confidence.topk(k, dim=1)
        self.states[context['scope']] = dict(
            sequence_ids=context['sequence_ids'], timestamp=context['timestamp'], pose=context['pose'],
            frame_index=context['frame_index'], confidence=scores.detach(),
            feature=gather_instances(feature.detach(), indices).clone(),
            anchor=gather_instances(anchor.detach(), indices).clone())
        if instance_ids is not None:
            self.states[context['scope']]['instance_ids'] = instance_ids.gather(1, indices).detach().clone()
        self.replays[context['scope']] = context
