import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

mnist_model = Net()
model_already_loaded = False

def load_mnist_model() -> bool:
    global mnist_model, model_already_loaded
    if not os.path.isfile("mnist_model/results/model.pth"):
        return False
    if model_already_loaded:
        return True
    
    network_state_dict = torch.load("mnist_model/results/model.pth")
    mnist_model = Net()
    mnist_model.load_state_dict(network_state_dict)

    model_already_loaded = True
    return True

def eval_mnist_on_file(img: Image.Image) -> int:
    mnist_model.eval()

    transform = transforms.transforms.Compose([
        transforms.transforms.Grayscale(1),
        transforms.transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]) 
    img_tensor = transform(img)
    
    with torch.no_grad():
        output = mnist_model(img_tensor)
            
    return output.data.max(1, keepdim=True)[1][0].item()