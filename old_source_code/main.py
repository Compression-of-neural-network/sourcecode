import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from torch.autograd.functional import hessian
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device('cpu' if torch.cuda.is_available() else "cpu")
CFG = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'epoch': 3,
    'lambda': 0.999
}

transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = MNIST('./data', train=True, download=True, transform=transformer)
test_data = MNIST('./data', train=False, download=True, transform=transformer)
train_loader = DataLoader(train_data, batch_size=CFG['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, batch_size=CFG['batch_size'], shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.fc1 = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
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

    def eval_hessian(self, loss_grad, model):
        cnt = 0
        for g in loss_grad:
            g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
            cnt = 1
        l = g_vector.size(0)
        hessian = torch.zeros(l, l)
        for idx in range(l):
            grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2
        return hessian

    def forward(self, x, y):
        cross_entropy = CrossEntropy(x, y)
        # calculate first order grad derivative of all weights
        first_grad = torch.autograd.grad(cross_entropy, net.parameters(), create_graph=True, retain_graph=True)
        # calculate second order grad derivative of every weight -- Hesse matrix diagonal
        second_grad = []
        for i, parm in enumerate(net.parameters()):
            second_grad.append(torch.autograd.grad(first_grad[i], parm, retain_graph=True, create_graph=True,
                                                   grad_outputs=torch.ones_like(first_grad[i]))[0])
        # calculate sum((d2L/dw2)^2)
        curvature = torch.tensor(0)
        for i in second_grad:
            curvature = curvature + torch.sum(torch.pow(i, 2))
        # calculate whole Hesse matrix
        hesse = self.eval_hessian(first_grad,net)
        (evals,evecs) = torch.eig(hesse,eigenvectors=True)
        #return cross_entropy * CFG['lambda'] + curvature * (1 - CFG['lambda']), cross_entropy, curvature
        curvature = torch.sum(hesse)
        return cross_entropy * CFG['lambda'] + curvature * (1 - CFG['lambda']), cross_entropy, curvature


if __name__ == '__main__':
    print(CFG)
    #log_csv = pd.read_csv('./log.csv')
    net = MyModel().to(device)
    optimizer = optim.Adam(net.parameters(), CFG['learning_rate'])
    Loss = []
    my_loss_function = MyLoss()
    CrossEntropy = nn.CrossEntropyLoss()
    for epoch in range(CFG['epoch']):
        train(epoch, net, Loss)
    test_loss, accuracy, crossentropy, curvature = test()
    '''log = pd.DataFrame(
        [[CFG['batch_size'], CFG['learning_rate'], CFG['epoch'], CFG['lambda'], round(test_loss, 6),
          round(crossentropy, 6), round(curvature,6), accuracy]],
        columns=['batch_size', 'learning_rate', 'epoch', 'lambda', 'test_loss', 'crossentropy', 'curvature',
                 'accuracy'])
    log_csv = log_csv.append(log, ignore_index=True)
    log_csv.to_csv('log.csv', index=False)'''
    '''    
    plt.plot(Loss)
    plt.xlabel('batch_idx')
    plt.ylabel('loss')
    plt.savefig('./lambda={}.jpg'.format(CFG['lambda']))'''
    # torch.save(net, 'lambda={}.pth'.format(CFG['lambda']))
