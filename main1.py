import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
CFG = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'epoch': 30,
    'lambda': 0.8
}

transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = CIFAR10('./data', train=True, download=True, transform=transformer)
test_data = CIFAR10('./data', train=False, download=True, transform=transformer)
train_loader = DataLoader(train_data, batch_size=CFG['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, batch_size=CFG['batch_size'], shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 3, padding=1)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.fc1 = nn.Linear(192 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(epoch, model, Loss):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = net.forward(data).to(device)
        loss, cross_entropy, curvature = my_loss_function(output, target)
        Loss.append(loss)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f} CrossEntropy: {:.6f} Curvature: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), cross_entropy, curvature))


def test():
    net.eval()
    test_loss = 0
    curvature = 0
    crossentropy = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data).to(device)
        t, cr, cu = my_loss_function(output, target)
        test_loss += t.item()
        crossentropy += cr.item()
        curvature += cu.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    crossentropy = crossentropy / len(test_loader)
    curvature = curvature / len(test_loader)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset), accuracy))
    return test_loss, accuracy, crossentropy, curvature


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        cross_entropy = CrossEntropy(x, y)
        firstgrad = torch.autograd.grad(cross_entropy, net.parameters(), create_graph=True, retain_graph=True)
        secondgrad = []
        for i, parm in enumerate(net.parameters()):
            secondgrad.append(torch.autograd.grad(firstgrad[i], parm, retain_graph=True, create_graph=True,
                                                  grad_outputs=torch.ones_like(firstgrad[i]))[0])
        curvature = torch.tensor(0)
        for i in secondgrad:
            curvature = curvature + torch.sum(torch.pow(i, 2))
        return cross_entropy * CFG['lambda'] + curvature * (1 - CFG['lambda']), cross_entropy, curvature


if __name__ == '__main__':
    print(CFG)
    log_csv = pd.read_csv('./log.csv')
    net = MyModel().to(device)
    optimizer = optim.Adam(net.parameters(), CFG['learning_rate'])
    Loss = []
    my_loss_function = MyLoss()
    CrossEntropy = nn.CrossEntropyLoss()
    for epoch in range(CFG['epoch']):
        train(epoch, net, Loss)
    test_loss, accuracy, crossentropy, curvature = test()
    log = pd.DataFrame(
        [[CFG['batch_size'], CFG['learning_rate'], CFG['epoch'], CFG['lambda'], round(test_loss, 6),
          round(crossentropy, 6), round(curvature,6), accuracy]],
        columns=['batch_size', 'learning_rate', 'epoch', 'lambda', 'test_loss', 'crossentropy', 'curvature',
                 'accuracy'])
    log_csv = log_csv.append(log, ignore_index=True)
    log_csv.to_csv('log.csv', index=False)
    torch.save(net, './labmda=08.pth')



