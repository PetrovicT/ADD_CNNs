from torch.utils.data import Dataset
from skimage.transform import rescale
import torch
import numpy as np


class AdniDataset(Dataset):
    def __init__(self, data, input_size):
        self.data = data
        self.input_size = input_size

    def __getitem__(self, index):
        image = np.load(self.data[index])
        image = rescale(image, self.input_size, anti_aliasing=False)
        target = int((1 if 'AD.npy' in self.data[index] else 0))
        return torch.tensor(image, dtype=torch.float32), target

    def __len__(self):
        return len(self.data)

