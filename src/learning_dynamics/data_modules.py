import lightning as L
import torch
from .transforms import PendulumTruncateTransform, ShallowWaterTruncateTransform
from .datasets import PendulumDataset, ShallowWaterDataset

class PendulumDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_path=None,
        test_data_path=None,
        batch_size=32,
        test_batch_size=32,
        val_split=0.6,
        num_workers=0
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.transform = PendulumTruncateTransform(self.val_split)

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = PendulumDataset(self.train_data_path, self.transform)
            self.val_dataset = PendulumDataset(self.train_data_path)
        elif stage == "test":
            self.test_dataset = PendulumDataset(self.test_data_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True
        )
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False
        )
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            num_workers=self.num_workers,
            batch_size=self.test_batch_size,
            shuffle=False
        )

class ShallowWaterDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_path=None,
        test_data_path=None,
        batch_size=32,
        test_batch_size=32,
        val_split=0.6,
        num_workers=0
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.transform = ShallowWaterTruncateTransform(self.val_split)

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = ShallowWaterDataset(self.train_data_path, self.transform)
            self.val_dataset = ShallowWaterDataset(self.train_data_path)
        elif stage == "test":
            self.test_dataset = ShallowWaterDataset(self.test_data_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True
        )
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False
        )
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            num_workers=self.num_workers,
            batch_size=self.test_batch_size,
            shuffle=False
        )