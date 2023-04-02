from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import torch
import numpy as np
import pandas as pd


class AdniDataset(Dataset):
    def __init__(self, data, input_size):
        #  self.data = pd.DataFrame(data)
        self.data = np.asarray(data)
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size, antialias=True)
        ])

    def __getitem__(self, index):
        file = self.data[index]
        #  file = self.data.iloc[index, 0]
        #  file = 'D:/preprocessed_OR/2d/AD_rgb/I31142_axial_25_AD.npy'
        image1 = torch.tensor(np.load(file).transpose(2, 1, 0), dtype=torch.float32)
        #  image = torch.tensor(np.load('./images\\I30550_coronal_37_CN.npy'), dtype=torch.float32)
        #  image = rescale(image, self.input_size, anti_aliasing=False)
        image = self.transform(image1)
        target = int((1 if 'AD.npy' in file else 0))
        #  target = int(1)
        return image, target

    def __len__(self):
        return len(self.data)

