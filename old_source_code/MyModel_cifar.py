import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


class MyModel_cifar(nn.Module):
    def __init__(self, CFG=None):
        super(MyModel_cifar, self).__init__()
        if CFG is None:
            CFG = {
                'batch_size': 128,
                'learning_rate': 0.001,
                'epoch': 200,
            }
        self.conv1 = nn.Conv2d(3, 48, 3, padding=1)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.fc1 = nn.Linear(192 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)

        self.CFG = CFG
        self.optimizer = optim.Adam(self.parameters(), self.CFG['learning_rate'])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        #self.train_data = MNIST('./data', train=True, download=True, transform=transformer)
        #self.test_data = MNIST('./data', train=False, download=True, transform=transformer)
        self.train_data = CIFAR10('./data', train=True, download=True, transform=transformer)
        self.test_data = CIFAR10('./data', train=False, download=True, transform=transformer)
        self.train_loader = DataLoader(self.train_data, batch_size=self.CFG['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.CFG['batch_size'], shuffle=True)
        self.Loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

    def MyTraining(self):
        for epoch in range(self.CFG['epoch']):
            running_loss = 0.0
            self.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.forward(data).to(self.device)
                loss = self.Loss_func(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f} '.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader),
                               running_loss / 100))

    def Mytest(self):
        self.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = self(data).to(self.device)
            t = self.Loss_func(output, target)
            test_loss += t.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                  len(self.test_loader.dataset),
                                                                                  accuracy))
        return test_loss, accuracy
