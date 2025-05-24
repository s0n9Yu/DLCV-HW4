import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

    
class OursDataset(Dataset):
    def __init__(self, path, patchsize=128, inference=False):
        super(OursDataset, self).__init__()
        self.path = path
        self.degraded_list = os.listdir(os.path.join(path, "degraded"))
        self.patchsize = patchsize
        self.inference = inference

    def __len__(self):
        return len(self.degraded_list)

    def __getitem__(self, index):

        if self.inference:
            image_name = self.degraded_list[index]
            img_degraded = Image.open(os.path.join(self.path, "degraded", image_name)).convert('RGB')
            img_degraded = ToTensor()(img_degraded)
            return img_degraded, image_name

        image_name = self.degraded_list[index]
        task_id = self.get_task_id(image_name)
        img_degraded = Image.open(os.path.join(self.path, "degraded", image_name)).convert('RGB')
        img_clean = Image.open(os.path.join(self.path, "clean", self.get_clean_name(image_name))).convert('RGB')
        img_degraded, img_clean = np.array(img_degraded), np.array(img_clean)

        img_degraded = crop_img(img_degraded, base=16)
        img_clean = crop_img(img_clean, base=16)
        degrad_patch, clean_patch = random_augmentation(*self._crop_patch(img_degraded, img_clean))

        clean_patch = ToTensor()(clean_patch)
        degrad_patch = ToTensor()(degrad_patch)


        return [image_name, task_id], degrad_patch, clean_patch


    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.patchsize)
        ind_W = random.randint(0, W - self.patchsize)

        patch_1 = img_1[ind_H:ind_H + self.patchsize, ind_W:ind_W + self.patchsize]
        patch_2 = img_2[ind_H:ind_H + self.patchsize, ind_W:ind_W + self.patchsize]

        return patch_1, patch_2

    @staticmethod
    def get_clean_name(name):
        name = name.replace("snow", "snow_clean")
        name = name.replace("rain", "rain_clean")
        return name
    @staticmethod
    def get_task_id(name):
        degradeId = {"snow": 0, "rain": 1}
        if "snow" in name:
            return degradeId["snow"]
        elif "rain" in name:
            return degradeId["rain"]
        return -1
