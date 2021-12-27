import os
from typing import Tuple, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split, DataLoader
from torchvision.models import vgg19_bn
from torchvision.transforms import transforms

from datatset import SuperResolutionDataset
from discriminator import Discriminator
from generator import Generator


class SuperResolutionDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 target_size: Tuple[int, int] = (1080, 1600),
                 batch_size: int = 64,
                 num_workers: int = 16
                 ):
        """
        Create datamodule for Super Resolution training
        :param root: dataset root directory
        :param batch_size:  batch size to use
        :param num_workers: number of workers for dataloaders
        """
        super().__init__()
        self.dataset_path = root
        self.train_set, self.val_set, self.test_set = None, None, None
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(size=target_size),
                transforms.ToTensor()
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.CenterCrop(size=target_size),
                transforms.ToTensor()
            ]
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        train_path = os.path.join(self.dataset_path, 'train')
        test_path = os.path.join(self.dataset_path, 'test')

        full_set = SuperResolutionDataset(train_path, transforms=self.train_transform)
        n_imgs = len(full_set)
        n_train = int(0.95 * n_imgs)
        self.train_set, self.val_set = random_split(full_set, [n_train, n_imgs - n_train])

        self.test_set = SuperResolutionDataset(test_path, transforms=self.test_transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class SuperResolutionGan(pl.LightningModule):
    def __init__(
            self,
            img_shape: Tuple[int, int],
            lr: float = 2e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            loss_weight: float = 10e-3
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator()
        self.discriminator = Discriminator(img_shape)
        # vgg features before last maxpool
        self.vgg = nn.Sequential(*vgg19_bn(pretrained=True).features[:-1])
        self.vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y, y_hat):
        return nnf.binary_cross_entropy(y, y_hat)

    def log_imgs(self, sample_imgs):
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, 0)

    def training_step(self, batch, batch_idx, optimizer_idx):
        lr_img, hr_img = batch
        if optimizer_idx == 0:
            valid = torch.ones(lr_img.size(0), 1, dtype=lr_img.dtype, device=self.device)
            g_img = self(lr_img)
            if batch_idx == 0:
                self.load_state_dict(g_img[:6])
            adv_loss = self.adversarial_loss(self.discriminator(g_img), valid)
            self.vgg.eval()
            feature_loss = nnf.mse_loss(self.vgg(g_img), self.vgg(hr_img))
            g_loss = feature_loss + self.hparams.loss_weight * adv_loss
            self.log('generator', {
                'adv_loss': adv_loss,
                'feature_loss': feature_loss,
                'g_loss': g_loss})
            return g_loss
        if optimizer_idx == 1:
            valid = torch.ones(lr_img.size(0), 1, dtype=lr_img.dtype, device=self.device)
            real_loss = self.adversarial_loss(self.discriminator(hr_img), valid)
            fake = torch.zeros(lr_img.size(0), 1, dtype=lr_img.dtype, device=self.device)
            fake_loss = self.adversarial_loss(self.discriminator(self(lr_img).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log('discriminator', {
                'real_loss': real_loss,
                'fake_loss': fake_loss,
                'd_loss': d_loss})
            return d_loss

    def validation_step(self, batch, batch_idx):
        lr_img, hr_img = batch
        g_img = self(lr_img)
        valid = torch.ones(lr_img.size(0), 1, dtype=lr_img.dtype, device=self.device)
        fake = torch.zeros(lr_img.size(0), 1, dtype=lr_img.dtype, device=self.device)
        adv_loss = self.adversarial_loss(self.discriminator(g_img), valid)
        feature_loss = nnf.mse_loss(self.vgg(g_img), self.vgg(hr_img))
        g_loss = feature_loss + self.hparams.loss_weight * adv_loss
        real_loss = self.adversarial_loss(self.discriminator(hr_img), valid)
        fake_loss = self.adversarial_loss(self.discriminator(self(lr_img).detach()), fake)

        self.log('generator', {
            'adv_loss': adv_loss,
            'feature_loss': feature_loss,
            'g_loss': g_loss})

        d_loss = (real_loss + fake_loss) / 2
        self.log('discriminator', {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'd_loss': d_loss})
        return g_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
