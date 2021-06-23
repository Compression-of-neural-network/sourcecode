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
        if CFG is None:
            CFG = {
                'batch_size': 128,
                'learning_rate': 0.001,
                'epoch': 200,
            }
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.fc1 = nn.Linear(16 * 4 * 4, 10)

        self.CFG = CFG
        self.optimizer = optim.Adam(self.parameters(), self.CFG['learning_rate'])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.train_data = MNIST('./data', train=True, download=True, transform=transformer)
        self.test_data = MNIST('./data', train=False, download=True, transform=transformer)
        self.train_loader = DataLoader(self.train_data, batch_size=self.CFG['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.CFG['batch_size'], shuffle=True)
        self.Loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


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


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    layer_set = [(1, 16, 1, 1),
                 (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                 (6, 32, 3, 2),
                 (6, 64, 4, 2),
                 (6, 96, 3, 1),
                 (6, 160, 3, 2),
                 (6, 320, 1, 1)]

    def __init__(self, num_classes=10, CFG=None):
        super(MobileNetV2, self).__init__()
        if CFG is None:
            CFG = {
                'batch_size': 128,
                'learning_rate': 0.001,
                'epoch': 200,
            }

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        self.CFG = CFG
        self.optimizer = optim.SGD(self.parameters(), lr=self.CFG['learning_rate'], momentum=0.9, weight_decay=5e-4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.train_data = CIFAR10('./data', train=True, download=True, transform=transformer)
        self.test_data = CIFAR10('./data', train=False, download=True, transform=transformer)
        self.train_loader = DataLoader(self.train_data, batch_size=self.CFG['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.CFG['batch_size'], shuffle=True)
        self.Loss_func = nn.CrossEntropyLoss()

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.layer_set:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


vgg_layer_set = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG16(nn.Module):
    layer_set = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    def __init__(self, CFG=None):
        super(VGG16, self).__init__()
        if CFG is None:
            CFG = {
                'batch_size': 128,
                'learning_rate': 0.001,
                'epoch': 200,
            }
        self.features = self._make_layers(self.layer_set)
        self.classifier = nn.Linear(512, 10)

        self.CFG = CFG
        self.optimizer = optim.SGD(self.parameters(), lr=self.CFG['learning_rate'], momentum=0.9, weight_decay=5e-4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.train_data = CIFAR10('./data', train=True, download=True, transform=transformer)
        self.test_data = CIFAR10('./data', train=False, download=True, transform=transformer)
        self.train_loader = DataLoader(self.train_data, batch_size=self.CFG['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.CFG['batch_size'], shuffle=True)
        self.Loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, layer_set):
        layers = []
        in_channels = 3
        for x in layer_set:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def MyTraining(model):
    model.train()
    for epoch in range(model.CFG['epoch']):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(model.train_loader):
            data = data.to(model.device)
            target = target.to(model.device)
            model.optimizer.zero_grad()
            output = model.forward(data).to(model.device)
            loss = model.Loss_func(output, target)
            loss.backward()
            model.optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f} '.format(
                    epoch, batch_idx * len(data), len(model.train_loader.dataset),
                           100. * batch_idx / len(model.train_loader),
                           running_loss / 100))


def Mytest(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in model.test_loader:
        data = data.to(model.device)
        target = target.to(model.device)
        output = model(data).to(model.device)
        t = model.Loss_func(output, target)
        test_loss += t.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(model.test_loader)
    accuracy = 100. * correct / len(model.test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                              len(model.test_loader.dataset), accuracy))
    return test_loss, accuracy
