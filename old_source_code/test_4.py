from MobileNetV2 import *
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
# from MyModel import *
from MyModel_MNIST import *
from MyModel_cifar import *
import operator
from kmeans_pytorch import kmeans_predict, kmeans
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_data = CIFAR10('./data', train=False, download=True, transform=transformer)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

#test_data = MNIST('./data', train=False, download=True, transform=transformer)
#test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
quantiz_level = [256, 128, 64, 32, 16, 8, 4]


def test(net):
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data).to(device)
        t = criterion(output, target)
        test_loss += t.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset),
                                                                              accuracy))


if __name__ == '__main__':
    net = MyModel_cifar().to(device)

    try:
        net.load_state_dict(torch.load('MyModel_cifar.mod'))
        # net = torch.load('./lambda=1.pth')
    except Exception:
        print('no such file')
    else:
        print('Successfully load net model')

    # --------------before kmeans--------------
    print('befrore Kmeans:')
    test(net)

    # ---------------kmeans----------------------

    parm = net.parameters()
    for i, p in enumerate(parm):

        if i == 0:
            weights = p.data.view(-1, 1)
        else:
            weights = torch.cat((weights, p.data.view(-1, 1)))
    # print(weights.size())

    cluster_ids_x, cluster_centers = kmeans(
        X=weights, num_clusters=4, distance='euclidean', device=device, tol=0.0005
    )

    # print(cluster_centers)

    parm_ = net.named_parameters()
    for name, param in parm_:
        # print(name, type(param))
        netweight = operator.attrgetter(name)(net)
        netweight_label = kmeans_predict(netweight.reshape(-1, 1), cluster_centers, 'euclidean', device=device)
        netweight_quanti = cluster_centers[netweight_label].reshape(netweight.size())
        # print(netweight_quanti)
        net_change = operator.attrgetter(name)(net)
        net_change.data.copy_(nn.parameter.Parameter(netweight_quanti.type(torch.cuda.FloatTensor)))
        # print(net_change.data)

    parm = net.parameters()
    for i, p in enumerate(parm):

        if i == 0:
            weights = p.data.view(-1, 1)
        else:
            weights = torch.cat((weights, p.data.view(-1, 1)))
    print(weights.data)

    # ------------------after kmeans---------------------
    print('after kmean:')
    test(net)

