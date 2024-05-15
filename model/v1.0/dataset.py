import torch
from torch.utils.data import Dataset
from utils import load_image
import os
import torchvision.transforms.v2 as v2

train_img_preprocessing = v2.Compose([
  v2.ToImage(), 
  v2.ToDtype(torch.float32, scale=True),
  v2.RandomHorizontalFlip(p=0.5),
  v2.RandomAdjustSharpness(sharpness_factor=2),
  v2.RandomAutocontrast(),
  v2.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])

base_img_preprocessing = v2.Compose([
  v2.ToImage(), 
  v2.ToDtype(torch.float32, scale=True),
  v2.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])

base_mask_preprocessing = v2.Compose([
  v2.ToImage(), 
  v2.ToDtype(torch.float32, scale=True),
  v2.Grayscale(num_output_channels=1)
])

class PlantsDataset(Dataset):

  def __init__(self,
               data: list,
               imgs_dir: str,
               masks_dir: str,
               preprocessing: v2.Compose = base_img_preprocessing):

    self.X_ids = []
    self.y_ids = []
    
    for x, y in data:
      path = os.path.join(imgs_dir, x)
      self.X_ids += [path]

      path = os.path.join(masks_dir, y)
      self.y_ids += [path]
      
    self.img_preprocessing = preprocessing
    self.mask_preprocessing = base_mask_preprocessing

  def __len__(self):
    return self.X_ids.__len__()

  def __getitem__(self, index):
    X = load_image(self.X_ids[index])
    y = load_image(self.y_ids[index])
    X = self.img_preprocessing(X)
    y = self.mask_preprocessing(y)
    return (X, y)


