from MyModel_cifar import MyModel_cifar
from MyModel_MNIST import MyModel_MNIST
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    net1 = MyModel_MNIST().to(device)
    net2 = MyModel_cifar().to(device)
    net1.MyTraining()
    net2.MyTraining()
    torch.save(net1.state_dict(), './MyModel_MNIST_200.mod')
    torch.save(net2.state_dict(), './MyModel_cifar_200.mod')
    net1.Mytest()
    net2.Mytest()
