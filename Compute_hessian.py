from MyModel import MyModel
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader
import png

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def eval_hessian(loss_grad, model):
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
    return hessian.cpu().data.numpy()


if __name__ == '__main__':
    net = MyModel().to(device)
    net.MyTraining()
    train_loader = DataLoader(net.train_data)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
    outputs = net.forward(data)
    loss = F.cross_entropy(outputs, target)
    loss_grad = torch.autograd.grad(loss, net.parameters(), create_graph=True)
    hessian = eval_hessian(loss_grad, net)
    pd.DataFrame(hessian).to_csv("./hessian.txt", header=None, index=None, sep=' ', mode='a')
