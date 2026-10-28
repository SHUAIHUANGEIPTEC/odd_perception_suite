from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from bev_vae.data.sequence_sampler import SequenceBatchSampler


class DataModuleFromConfig(LightningDataModule):
    def __init__(
        self, 
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        drop_last: bool = True,
        temporal_sampling: bool = False,
        sampler_seed: int = 12345,
        train: Optional[Dataset] = None,
        val: Optional[Dataset] = None,
        test: Optional[Dataset] = None,
        ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):
        if self.hparams.train is not None:
            print(f'Train dataset has {len(self.hparams.train)} samples')
        if self.hparams.val is not None:
            print(f'Val dataset has {len(self.hparams.val)} samples')
        if self.hparams.test is not None:
            print(f'Test dataset has {len(self.hparams.test)} samples')

    def _temporal_dataloader(self, dataset, batch_size, shuffle):
        if not getattr(dataset, "temporal", False):
            raise ValueError("temporal_sampling requires a temporal dataset")
        trainer = self.trainer
        # Lightning 2.1 stores this constructor option on its connector.
        if trainer._accelerator_connector.use_distributed_sampler:
            raise ValueError("Temporal sampling requires trainer.use_distributed_sampler=false")
        sampler = SequenceBatchSampler(dataset, batch_size, rank=trainer.global_rank,
            world_size=trainer.world_size, seed=self.hparams.sampler_seed, shuffle=shuffle)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def train_dataloader(self):
        if self.hparams.temporal_sampling:
            return self._temporal_dataloader(self.hparams.train, self.hparams.train_batch_size, True)
        return DataLoader(
            dataset=self.hparams.train, 
            batch_size=self.hparams.train_batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory, 
            shuffle=self.hparams.shuffle, 
            drop_last=self.hparams.drop_last)

    def val_dataloader(self):
        if self.hparams.temporal_sampling:
            return self._temporal_dataloader(self.hparams.val, self.hparams.val_batch_size, False)
        return DataLoader(
            dataset=self.hparams.val, 
            batch_size=self.hparams.val_batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory, 
            shuffle=False)

    def test_dataloader(self):
        if self.hparams.temporal_sampling:
            return self._temporal_dataloader(self.hparams.test, self.hparams.test_batch_size, False)
        return DataLoader(
            dataset=self.hparams.test, 
            batch_size=self.hparams.test_batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory, 
            shuffle=False)
