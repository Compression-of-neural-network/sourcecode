import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


class MyModel_MNIST(nn.Module):
    def __init__(self, CFG=None):
        super(MyModel_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.fc1 = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output
