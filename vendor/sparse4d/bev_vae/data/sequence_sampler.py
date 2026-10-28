"""Rank-sharded scene streams, with one chronological stream per batch slot.

An epoch is a sample-exposure budget, as in Sparse4D's streaming sampler;
short lanes cycle, rather than padding a batch with fake frames. No random
frame replacement, frame reversal or implicit DistributedSampler is allowed.
"""
import math
import numpy as np
from torch.utils.data import Sampler


class SequenceBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, rank=0, world_size=1, seed=0, shuffle=True):
        self.groups = dataset.sequence_groups
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        global_batch = self.batch_size * self.world_size
        if not 0 <= self.rank < self.world_size or self.batch_size < 1:
            raise ValueError('Invalid temporal sampler rank/world size/batch size')
        if len(self.groups) < global_batch or any(len(group) == 0 for group in self.groups):
            raise ValueError('Temporal sampling needs at least one nonempty sequence per global batch slot')
        self.num_batches = math.ceil(sum(map(len, self.groups)) / global_batch)

    def __len__(self):
        return self.num_batches

    @property
    def sampler(self):
        # Lightning calls batch_sampler.sampler.set_epoch before advancing epochs.
        return self

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        order = rng.permutation(len(self.groups)) if self.shuffle else np.arange(len(self.groups))
        global_batch = self.batch_size * self.world_size
        lanes = []
        for slot in range(self.batch_size):
            global_slot = self.rank * self.batch_size + slot
            # Disjoint ownership for the entire epoch, across ranks AND slots.
            lane_groups = order[global_slot::global_batch]
            lane_rng = np.random.RandomState((self.seed + self.epoch * 100003 + global_slot) % (2**32))
            lanes.append(self._lane(lane_groups, lane_rng, global_slot))
        for _ in range(self.num_batches):
            yield [next(lane) for lane in lanes]

    def _lane(self, group_ids, rng, global_slot):
        cycle = 0
        while True:
            for group_id in group_ids:
                aug_seed = int(rng.randint(0, 2**31 - 1))
                sequence_id = f'{self.epoch}:{global_slot}:{cycle}:{int(group_id)}'
                for offset, index in enumerate(self.groups[int(group_id)]):
                    yield dict(idx=int(index), aug_seed=aug_seed, sequence_id=sequence_id,
                               sequence_start=offset == 0)
            cycle += 1
