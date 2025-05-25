import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from net.model import PromptIR
import lightning.pytorch as pl

from utils.dataset_ours import OursDataset

trainset = OursDataset(path="data_ours/train", patchsize=256)


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        # lr = scheduler.get_lr()

    def configure_optimizers(self):
        ...
        '''
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=150
        )
        '''
        # return [optimizer],[scheduler]


parser = argparse.ArgumentParser()
# Input Parameters
parser.add_argument(
    '--ckpt_path',
    type=str,
    default="model.ckpt",
    help='checkpoint save path'
)
opt = parser.parse_args()

net = PromptIRModel.load_from_checkpoint(opt.ckpt_path)
net = net.cuda()
net.eval()

dataset = OursDataset(path="data_ours/test", inference=True)

testloader = DataLoader(
    dataset,
    batch_size=1,
    pin_memory=True,
    shuffle=False,
    num_workers=0
)
img_dict = {}
with torch.no_grad():
    for (degrad_patch, image_name) in tqdm(testloader):
        print(image_name[0])
        print("input max", degrad_patch.max())
        degrad_patch = degrad_patch.cuda()
        restored = net(degrad_patch).squeeze(0).cpu().numpy()
        # Rearrange to (3, H, W)
        # img_array = np.transpose(restored, (2, 0, 1))
        img_array = restored
        img_array = np.clip(img_array, a_min=0, a_max=1)
        img_array = img_array * 255
        img_array = img_array.astype(np.uint8)

        print(img_array.shape)
        print("max ", img_array.max())
        print("min ", img_array.min())
        print("avg ", img_array.mean())

        '''
        img_hwc = np.transpose(img_array, (1, 2, 0))
        # Display the image
        plt.imshow(img_hwc)
        plt.axis('off')  # Optional: turn off axis
        plt.show()
        '''

        # Add to dictionary
        img_dict[image_name[0]] = img_array

# Save to .npz file
output_npz = 'pred.npz'
np.savez(output_npz, **img_dict)

print(f"Saved {len(img_dict)} images to {output_npz}")
