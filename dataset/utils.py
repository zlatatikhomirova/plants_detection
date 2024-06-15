import numpy as np
from albumentations import (CenterCrop, RandomRotate90, 
                            GridDistortion, HorizontalFlip, 
                            VerticalFlip)
import cv2
from objects import Sample
import torch
from torchvision.utils import (
            draw_segmentation_masks, 
            draw_bounding_boxes) 
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes
import matplotlib.pyplot as plt
import random
import os

plt.rcParams["savefig.bbox"] = "tight"
SUPPORTED_IMAGE_FORMATS = ("jpg", "png", "tif", "jpeg", "svg")

def norm_denoise(img: np.ndarray) -> np.ndarray:
  """
  Убирает шум и нормализует изображение.
  img - изображение в формате np.ndarray
  """
  img_normalized = cv2.normalize(img, np.zeros((img.shape[0], 
                                                img.shape[1])),
                                  0, 255, cv2.NORM_MINMAX)
  img_denoised = cv2.fastNlMeansDenoising(img_normalized,
                                          None, 20, 7, 15)
  return img_denoised

def load_image(path: str) -> np.ndarray | None:
  """
  Загружает изображение из файла, если расшираение файла
  входит в список SUPPORTED_IMAGE_FORMATS.
  path - путь к изображению
  """
  halfpath, sep, ext = path.rpartition(".")
  if ext.lower() in SUPPORTED_IMAGE_FORMATS:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
  raise ValueError(f"Недопустимое расширение файла: .{ext}")

def augment_item(sample, aug_function):
  """
  
  sample - объект Sample
  aug_function - функция из albumentations
  """
  augmented = aug_function(image=sample.image, mask=sample.mask)
  return augmented["image"], augmented["mask"]
  
def augment_data(samples, crop_coef=0.8):
  
  H = int(samples[0].image.shape[1] * crop_coef)
  W = int(samples[0].image.shape[2] * crop_coef)
  
  augmented = []  
  
  for sample in samples:
    img, mask = augment_item(sample, CenterCrop(H, W, p=1.0))
    img = cv2.resize(img, sample.image.shape[1:3])
    mask = cv2.resize(mask, sample.image.shape[1:3])
    augmented += [Sample(img, mask)]
    augmented += [Sample(*augment_item(sample,
                         RandomRotate90(p=1.0)))]
    augmented += [Sample(*augment_item(sample,
                          GridDistortion(p=1.0)))]
    augmented += [Sample(*augment_item(sample,
                          HorizontalFlip(p=1.0)))]
    augmented += [Sample(*augment_item(sample,
                          VerticalFlip(p=1.0)))]
  return samples + augmented

def img_to_bin_colors(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.inRange(img, 200, 255) # 200 - выбрано экспериментально 
    return img

def computing_contours(mask: np.ndarray) -> tuple:
    """
        получение кортежа контуров по маске
        
        return 
            typle(np.ndarray)
    """
    
    if (not isinstance(mask, np.ndarray)):
        mask = np.asarray(mask)
        
    if (mask.ndim != 2):
        mask = img_to_bin_colors(mask)
        
    cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    return cnts

def calc_sfs(img: np.array) -> int:
    cnts = computing_contours(img)
    return len(cnts[0])

def draw_contours(result_img: np.ndarray | torch.Tensor, cnts: tuple ,threshold_value_plt:int = 20, color_border: int | tuple = (0, 0, 255)) -> torch.Tensor:
    """_summary_
        Насенение bb на копию result_img
    Args:
        result_img (np.ndarray) - изображение, на копию которого будут нанесены bb
        threshold_value_plt- пороговое значение площади подсолнуха, который будет помещен в bb (все, что больше будут отмечены)

     Returns:
        torch.Tensor: изображение result_img с нанесенными bb
        
    """
    
    tmp_result_img = result_img.copy()
    
    # проверка типов
    if (not isinstance(tmp_result_img, np.ndarray)):
        tmp_result_img = np.asarray(tmp_result_img)
    
    # проверка количества каналов
    if(tmp_result_img.ndim == 2):
        color_border = 100
        print("На вход подано двухканальное изображение. Цвет bb - серый.")

    # отрисовка bb
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if ((x+w)*(y+h) > threshold_value_plt):
            cv2.rectangle(tmp_result_img, (x, y), (x+w, y+h), color_border, 2)

    return torch.from_numpy(tmp_result_img)

def show(imgs, title=''):
  """
  Выводит изображения. Работа с изображениями, тип которых
  torch.Tensor.
  imgs -  некоторые изображения.
  """
  if not isinstance(imgs, list):
      imgs = [imgs]
  fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
  for i, img in enumerate(imgs):
      img = img.detach()
      img = F.to_pil_image(img)
      axs[0, i].imshow(np.asarray(img))
      axs[0, i].set(xticklabels=[], yticklabels=[],
                    xticks=[], yticks=[])
  plt.title(title)
  plt.show()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
