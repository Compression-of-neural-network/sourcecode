#from MyModel_cifar import MyModel_cifar
#from MyModel_MNIST import MyModel_MNIST
from Models import *
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = CIFAR10('./data', train=True, download=True, transform=transformer)
test_data = CIFAR10('./data', train=False, download=True, transform=transformer)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    net1 = VGG16().to(device)
    #net2 = MyModel_cifar().to(device)
    #MyTraining(net1)
    #MyTraining(net2)
    #torch.save(net1.state_dict(), './VGG16_200.mod')
    #torch.save(net2.state_dict(), './MyModel_cifar_200.mod')
    #Mytest(net1)
    #Mytest(net2)

    net1.load_state_dict(torch.load('./VGG16_200.mod'))
    #accuracy = 100. * correct / len(test_loader.dataset)
    Mytest(net1)
    net1.eval()
    test_loss = 0
    correct = 0
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        output = net1(data).to(device)

        t = criterion(output, target)
        test_loss += t.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                              len(train_loader.dataset),
                                                                              accuracy))