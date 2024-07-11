import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './', batch_size: int = 32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.mnist_train = MNIST(self.data_dir, train=True, download=True)
        self.mnist_test = MNIST(self.data_dir, train=False, download=True)
        self.mnist_trian, self.mnist_val = torch.utils.data.random_split(self.mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

