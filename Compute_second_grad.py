from MobileNetV2 import *
from MyModel import *

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = CIFAR10('./data', train=True, download=True, transform=transformer)

#transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#train_data = MNIST('./data', train=True, download=True, transform=transformer)


def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    print(l)
    hessian = torch.zeros(l, 1)
    for idx in range(l):
        grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        #print(idx)
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            #print(g2.data.size())
            cnt = 1
        hessian[idx] = g2[idx]
    return hessian.cpu().data.numpy()


def compute_hessian(net):
    train_loader = DataLoader(train_data)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
    outputs = net.forward(data)
    loss = F.cross_entropy(outputs, target)

    first_grad = torch.autograd.grad(loss, net.parameters(), create_graph=True, retain_graph=True)
    # calculate second order grad derivative of every weight -- Hesse matrix diagonal
    for i, parm in enumerate(net.parameters()):
        print(i)
        if i == 0:
            second_grad = torch.autograd.grad(first_grad[i], parm, retain_graph=True, create_graph=True,
                                               grad_outputs=torch.ones_like(first_grad[i]))[0].data.view(-1, 1)
        else:
            se_grad = torch.autograd.grad(first_grad[i], parm, retain_graph=True, create_graph=True,
                                               grad_outputs=torch.ones_like(first_grad[i]))[0].data.view(-1, 1)
            second_grad = torch.cat([second_grad, se_grad])

    #loss_grad = torch.autograd.grad(loss, net.parameters(), create_graph=True, retain_graph=True)
    #hessian = eval_hessian(loss_grad, net)
    #loss_grad_2 = torch.autograd.grad(loss_grad, net.parameters(), create_graph=True)
    return second_grad
    # first_order_grads = torch.autograd.grad(loss, net.parameters(), retain_graph=True, create_graph=True)
    # first_order_grads = torch.cat([x.view(-1, 1) for x in first_order_grads])
    # print(first_order_grads.data.size())
    # second_order_grads = []
    # for idx, first_grad in enumerate(first_order_grads):
    #     print('\nfirst_grad_id: %d' % idx)
    #     grad2rd = torch.autograd.grad(first_grad, net.parameters(), create_graph=True)
    #     cnt = 0
    #     for g in grad2rd:
    #         g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
    #         cnt = 1
    #     second_order_grads.append(g2)
    # parm = net.parameters()
    # for i, p in enumerate(parm):
    #
    #     if i == 0:
    #         weights = p.data.view(-1, 1)
    #     else:
    #         weights = torch.cat((weights, p.data.view(-1, 1)))
    # print(weights.size())
    #
    # for grad_id, grads in enumerate(first_order_grads):
    #     print('\nfirst_grad_id: %d' % grad_id)
    #     s_grads = torch.autograd.grad(grads, weights, retain_graph=True, allow_unused=True)
    #     second_order_grads.append(s_grads)

    # return second_order_grads


if __name__ == '__main__':

    #net = MyModel().to(device)
    net = MobileNetV2().to(device)
    try:
        net.load_state_dict(torch.load('./MobileNetV2_200.mod'))
    except Exception:
        print('no such file')
    else:
        print('Successfully load net model')
    hessian = compute_hessian(net)
    #print(hessian.shape)
    print(hessian.data.size())
    torch.save(hessian, 'second_grad.pt')
