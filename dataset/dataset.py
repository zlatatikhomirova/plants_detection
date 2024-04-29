import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import augment_data, norm_denoise

class PlantsDataset(Dataset):

    def __init__(self, data,
                need_preprocessing: bool = True):
        
        data = augment_data(data)

        X = [sample.image for sample in data]
        if need_preprocessing:
          X = [norm_denoise(x) for x in X]
        X = np.array(X)
        y = np.array(
                 [sample.mask for sample in data]
                    )
        
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
      return (self.X[index], self.y[index])  


              