import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
#import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.dataset_ours import OursDataset


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        self.l2loss = nn.MSELoss()
        self.val_outputs = []
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = (self.loss_fn(restored,clean_patch) + self.l2loss(restored, clean_patch)) / 2
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = nn.MSELoss()(restored,clean_patch)
        #self.log("val_loss", loss, prog_bar=True)
        self.val_outputs.append(loss)  # store for epoch-end processing

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.val_outputs).mean()
        self.log("val_loss", avg_val_loss, prog_bar=True, sync_dist=True)
        self.val_outputs.clear()  # reset for next epoch

    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        #scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=500, eta_min=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

        return [optimizer],[scheduler]






def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    #trainset = PromptTrainDataset(opt)
    trainset = OursDataset(path="data_ours/train", patchsize=128)
    validset = OursDataset(path="data_ours/valid", patchsize=128)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    validloader = DataLoader(validset, batch_size=opt.batch_size, pin_memory=True, shuffle=False,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = PromptIRModel()
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=1,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=validloader)


if __name__ == '__main__':
    main()



