from MyModel import MyModel
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    net = MyModel().to(device)
    net.MyTraining()
    torch.save(net.state_dict(), './MyModel.mod')