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

def make_logical_masks_and_bboxes(mask):
  """
  Поучить логические маски для поля.
  """
  if isinstance(mask, np.ndarray):
    mask = torch.fromnumpy(mask)
  # уникальные цвета масок
  obj_ids = torch.unique(mask)
  # первый из уникальных цветов - фон, удаляем это значение
  obj_ids = obj_ids[1:]
  # разделим маску с цветовой кодировкой на набор логических масок.
  masks = mask == obj_ids[:, None, None]
  return masks, masks_to_boxes(masks) 

def draw_seg_masks(img, mask, logical_masks,
                   alpha=0.8, colors='blue'):
  """
  Рисует маски растительности на изображении.
  """
  if isinstance(img, np.ndarray):
    img = torch.fromnumpy(img)
  if isinstance(mask, np.ndarray):
    mask = torch.fromnumpy(mask)
  drawn_masks = []
  for mask in logical_masks:
    drawn_masks.append(draw_segmentation_masks(img, mask, 
                                               alpha=alpha, 
                                               colors=colors))
  show(drawn_masks, title='Segmentation masks on image')

def draw_bboxes(img, bboxes_list, colors="red"):
  """
  Рисует bounding boxes на изображении.
  """
  if isinstance(img, np.ndarray):
    img = torch.fromnumpy(img)
  drawn_boxes = draw_bounding_boxes(img, 
                                    bboxes_list,
                                    colors=colors)
  show(drawn_boxes, title='Bounding boxes on image')
  
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