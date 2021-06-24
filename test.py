import torch
from Models import *
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    net = VGG16().to(device)
    net.load_state_dict(torch.load('./saved_models_after_training/VGG16_200.mod'))
    parm = net.parameters()
    weights = parm.__next__().view(-1)
    for i, n in enumerate(parm):
        if i == 0:
            continue
        else:
            weights = torch.cat([weights, n.view(-1)])
    print(weights.size())

