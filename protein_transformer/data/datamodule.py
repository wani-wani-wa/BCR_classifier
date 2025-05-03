import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import BCRDataset


class BCRDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, tokenizer, batch_size=32, max_length=320):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = BCRDataset(self.train_df, self.tokenizer, self.max_length)
            self.val_dataset = BCRDataset(self.val_df, self.tokenizer, self.max_length)
        if stage in (None, "test"):
            self.test_dataset = BCRDataset(self.test_df, self.tokenizer, self.max_length)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
