import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import cv2
import numpy as np
import seaborn as sns
from random import choice, seed, randint

from agrocv_model.dataseta import base_img_preprocessing
from agrocv_model.objects import Field
from agrocv_model.utils import load_image

from backbones_unet.model.unet import Unet
from backbones_unet.utils.trainer import Trainer

def activate_model(path, device):
    model = Unet(
        backbone='convnext_base',
        in_channels=3,
        num_classes=1,
        )
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.to(device)
    return model

def predict(model, data):
    global device
    model.eval()  # testing mode
    with torch.no_grad():
        data = data.to(device)
        pred = model(data)
        pred = (pred > 0).type(torch.uint8)
        return pred
    
def count_and_cut(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    threshold = 0.5 * np.mean(areas)
    contours = [cnt for i, cnt in enumerate(contours) if areas[i] < threshold]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img, len(contours), contours

def get_mask(path_to_field,
             model,
             device,
             result_path):

    paths = []

    if not os.path.exists(path_to_field):
        raise ValueError("Path do not exist.")

    print("Загружаем фотографии...")
    # загрузка изображений
    if os.path.isfile(path_to_field):
        paths += [path_to_field]
    else:
        for filename in os.listdir(path_to_field):
            img_path = os.path.join(path_to_field, filename)
            if os.path.isfile(img_path):
                paths += [img_path]

    for i, path in enumerate(paths):
        #print(f"Обрабатываем изображения полей: {i + 1}/{len(paths)}")
        img = load_image(path)
        field = Field(img, stride = (192, 192))
        #print("Нарезаем изображение на фрагменты...")
        field.crop2sample()
        for j, sample in enumerate(field.samples):
            #print(f"\rОбрабатываем фрагменты: {j + 1}/{len(field.samples)}")
            X = base_img_preprocessing(sample.image).to(device)
            sample.mask = predict(model, X.unsqueeze(0)).squeeze(0, 1).cpu().numpy()
        #print("Создаём маску поля...")
        field.samples2union()
        f = np.where(field.mask > 0, 255, 0)
        f = f.astype(np.uint8)
        # img, num_plants, _ = count_and_cut(f)
        # imgname, sep, ext = path.split("/")[-1].partition(".")
        # newname = f"{imgname}_num_{num_plants}.{ext}"
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
        path_full = os.path.join(result_path, path.split("/")[-1])
        cv2.imwrite(path_full, f)
        print(f"Сохранено в {path_full}")
        #plt.imshow(f)
        #plt.show()
        return f

device = None
model = None
agrocv_initialized = False

def agrocv_model_eval(path):
    global device, model, agrocv_initialized
    if not agrocv_initialized:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = activate_model("agrocv_model/model_v0.4_0.5019_vloss0.0000_score.pth", device)
        agrocv_initialized = True

    params = {'path_to_field' : path,
          'model': model,
          'device' : device,
          'result_path' : "public/results",}

    get_mask(**params)