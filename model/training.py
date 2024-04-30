import os
from objects import Field, Sample
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from dataset import PlantsDataset
from model import UNet, predict
import matplotlib.pyplot as plt
from trainer import Trainer
from torch.utils.data import DataLoader
from utils import (
                load_image, seed_everything,
                make_logical_masks_and_bboxes,
                draw_seg_masks,  draw_bboxes
                  )

SEED = 42

seed_everything(42)

PATH_TO_IMGS = "/content/drive/MyDrive/plants_detection/img"
PATH_TO_MASKS = "/content/drive/MyDrive/plants_detection/mask"

imgs_names = os.listdir(PATH_TO_IMGS)
masks_names = os.listdir(PATH_TO_MASKS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fields = []
for filename in imgs_names:
    name, _, ext = filename.rpartition('.')
    mask_name = name + "_mask." + ext
    if not mask_name in masks_names:
      raise ValueError("no mask name for this image.")
    path_to_img = os.path.join(PATH_TO_IMGS, filename)
    path_to_mask = os.path.join(PATH_TO_MASKS, mask_name)
    image = load_image(path_to_img)
    mask = load_image(path_to_mask)
    if image is None or mask is None:
      continue
    field = Field(image, mask)
    field.crop2sample()
    fields.append(field)

data = []
for field in fields:
  data += field.samples
  
data_train, data_test = train_test_split(data, test_size=0.2, 
                                   random_state=SEED
                                   )
data_val, data_test = train_test_split(data_test, test_size=0.5,
                                  random_state=SEED
                                )


model = UNet(3, 1).to(device)

params = {
    'dataset': PlantsDataset,
    'net': model,
    'epoch_amount': 1000, 
    'learning_rate': 1e-2,
    'early_stopping': 25,
    'loss_f': nn.BCELoss(),
    'optim': torch.optim.SGD,
}

clf = Trainer(**params)
clf.fit(data_train, data_val)

test = PlantsDataset(data_test)
test_dataloader = DataLoader(test,
                  batch_size=clf.batch_size, shuffle=False)

test_loss, test_metric = clf.test(clf.best_model, test_dataloader)
print("Test loss", test_loss)
print("Test metric", test_metric)

# создадим отчет об обучении модели
PATH_TO_REPORT = "/content/stats.txt"
clf.make_report(PATH_TO_REPORT)

# графики лосса и метрики в течение обучения
clf.plot_loss()
clf.plot_metric()

# создадим маску и bounding box-ы для тестового изображения
image, real_mask = test[0]
pred_mask = predict(clf.best_model, image)
logical_masks, bboxes = make_logical_masks_and_bboxes(pred_mask)

plt.imshow(image) 
plt.title('Source image') 
plt.show() 

plt.imshow(real_mask) 
plt.title('Real mask') 
plt.show() 

plt.imshow(pred_mask)
plt.title('Predicted mask') 
plt.show() 

draw_seg_masks(image, pred_mask, logical_masks)
draw_bboxes(image, bboxes)

MODEL_SAVE_PATH = f"/content/model_m{clf.best_metric:.2f}_{clf.best_val_loss:.2f}.pth"

# сохраняем модель
torch.save(obj=clf.best_model.state_dict(), # сохраняем
           f=MODEL_SAVE_PATH)  # параметры модели