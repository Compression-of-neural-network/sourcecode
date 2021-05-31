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
from tensorboardX import SummaryWriter

import argparse, os
#Guard protection for error: No module named 'tkinter'
try: import matplotlib.pyplot as plt
except ModuleNotFoundError:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
from utils import normal_dist, expected_normal_dist, MSE_loss, LloydMaxQuantizer

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
CFG = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'epoch': 3,
    'lambda': 1
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


def test(net):
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


if __name__ == '__main__':
    net = torch.load('labmda=1.pth')
    # calculate mean and var of weights
    parm = list(net.parameters())
    weights = parm[0].view(-1)
    for i, p in enumerate(parm):
        if i == 0:
            continue
        if i % 2 == 0:
            weights = torch.cat([weights, p.view(-1)])
    weights_mean = torch.mean(weights)
    weights_var = torch.var(weights)


    bit = 4
    iteration = 100
    x = weights.to('cpu').detach().numpy()
    repre = LloydMaxQuantizer.start_repre(x, bit)
    min_loss = 1.0
    for i in range(iteration):
        thre = LloydMaxQuantizer.threshold(repre)
        # In case wanting to use with another mean or variance, need to change mean and variance in untils.py file
        repre = LloydMaxQuantizer.represent(thre, expected_normal_dist, normal_dist)
        x_hat_q = LloydMaxQuantizer.quant(x, thre, repre)
        loss = MSE_loss(x, x_hat_q)

        # Print every 10 loops
        if (i % 10 == 0 and i != 0):
            print('iteration: ' + str(i))
            print('thre: ' + str(thre))
            print('repre: ' + str(repre))
            print('loss: ' + str(loss))
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        # Keep the threhold and representation that has the lowest MSE loss.
        if (min_loss > loss):
            min_loss = loss
            min_thre = thre
            min_repre = repre

    print('min loss' + str(min_loss))
    print('min thre' + str(min_thre))
    print('min repre' + str(min_repre))

    index = []
    for i in range(2**bit):
        index.append(np.int8(i))
    codebook = dict(zip(min_repre,index))



    weights_conv1 = net.conv1.weight.to('cpu').detach().numpy()
    Q_weights_conv1 = LloydMaxQuantizer.quant(weights_conv1, min_thre, min_repre)
    T_weights_conv1 = torch.from_numpy(Q_weights_conv1)
    net.conv1.weight = nn.parameter.Parameter(T_weights_conv1.type(torch.cuda.FloatTensor))

    weights_conv2 = net.conv2.weight.to('cpu').detach().numpy()
    Q_weights_conv2 = LloydMaxQuantizer.quant(weights_conv2, min_thre, min_repre)
    T_weights_conv2 = torch.from_numpy(Q_weights_conv2)
    net.conv2.weight = nn.parameter.Parameter(T_weights_conv2.type(torch.cuda.FloatTensor))

    weights_conv3 = net.conv3.weight.to('cpu').detach().numpy()
    Q_weights_conv3 = LloydMaxQuantizer.quant(weights_conv3, min_thre, min_repre)
    T_weights_conv3 = torch.from_numpy(Q_weights_conv3)
    net.conv3.weight = nn.parameter.Parameter(T_weights_conv3.type(torch.cuda.FloatTensor))

    weights_fc1 = net.fc1.weight.to('cpu').detach().numpy()
    Q_weights_fc1 = LloydMaxQuantizer.quant(weights_fc1, min_thre, min_repre)
    T_weights_fc1 = torch.from_numpy(Q_weights_fc1)
    net.fc1.weight = nn.parameter.Parameter(T_weights_fc1.type(torch.cuda.FloatTensor))

    weights_fc2 = net.fc2.weight.to('cpu').detach().numpy()
    Q_weights_fc2 = LloydMaxQuantizer.quant(weights_fc2, min_thre, min_repre)
    T_weights_fc2 = torch.from_numpy(Q_weights_fc2)
    net.fc2.weight = nn.parameter.Parameter(T_weights_fc2.type(torch.cuda.FloatTensor))

    weights_fc3 = net.fc3.weight.to('cpu').detach().numpy()
    Q_weights_fc3 = LloydMaxQuantizer.quant(weights_fc3, min_thre, min_repre)
    T_weights_fc3 = torch.from_numpy(Q_weights_fc3)
    net.fc3.weight = nn.parameter.Parameter(T_weights_fc3.type(torch.cuda.FloatTensor))

    my_loss_function = MyLoss()
    CrossEntropy = nn.CrossEntropyLoss()
    test(net)
    torch.save(net, './labmda=2_quantized')





