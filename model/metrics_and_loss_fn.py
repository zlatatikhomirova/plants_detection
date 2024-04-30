import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2

def iou_pytorch(outputs: torch.Tensor,
                labels: torch.Tensor):
  # передавая выходные данные из UNet или чего-то еще, скорее всего, это будет
  # в форме BATCH x 1 x В x Ш
  # BATCH x 1 x H x W => BATCH x H x W
  outputs = outputs.squeeze(1).byte()  
  labels = labels.squeeze(1).byte()
  SMOOTH = 1e-8
  # будет равно нулю, если реальыне=0 или предсказание=0
  intersection = (outputs & labels).float().sum((1, 2))
  # 0 если оба 0
  union = (outputs | labels).float().sum((1, 2))
  # защита от деления на 0
  iou = (intersection + SMOOTH) / (union + SMOOTH)
  thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10 
  return thresholded

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
      
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
      
        Dice_BCE = BCE + dice_loss
      
        return Dice_BCE

class IoULoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
      super(IoULoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
      inputs = F.sigmoid(inputs)
      #flatten label and prediction tensors
      inputs = inputs.view(-1)
      targets = targets.view(-1)
      #intersection is equivalent to True Positive count
      #union is the mutually inclusive area of all labels & predictions
      intersection = (inputs * targets).sum()
      total = (inputs + targets).sum()
      union = total - intersection

      IoU = (intersection + smooth)/(union + smooth)

      return 1 - IoU

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA,
                gamma=GAMMA, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets,
                                     reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        return focal_loss

