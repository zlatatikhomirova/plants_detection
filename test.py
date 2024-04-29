from model import UNet, predict
import os
from objects import Field
import torch
from utils import make_logical_masks_and_bboxes, load_image
# img -> field -> pred samples -> union -> find bboxes and count (visualize)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mask(path_to_field, model_path, device):
  model = UNet()
  model.load_state_dict(torch.load(model_path))
  model.to(device)
  model.eval()

  fields = []
  
  if not os.path.exists(path_to_field):
    raise ValueError("Incorrect path.")

  # загрузка изображений
  if os.path.isfile(path_to_field):
    img = load_image(path_to_field)
    field = Field(img)
    fields = [field]
  else:
    for filename in os.listdir(path_to_field):
      img_path = os.path.join(path_to_field, filename)
      if os.path.isfile(img_path):
        img = load_image(img_path)
        fields += [Field(img)]

  for field in fields:
    field.crop2sample()
    for sample in field.samples:
        X = sample.image.to(device)
        sample.mask = predict(model, X)
    field.samples2union()
    field.bboxes, field.logical_masks = make_logical_masks_and_bboxes(field.mask)
    field.count_plants()
  return fields


