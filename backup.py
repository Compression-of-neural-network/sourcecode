from MobileNetV2 import *
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
#from MyModel import *
from KaiModel import *
from kmeans_pytorch import kmeans, kmeans_predict
import operator


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_data = CIFAR10('./data', train=False, download=True, transform=transformer)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()


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
    #net = MyModel().to(device)
    net = MobileNetV2().to(device)
    # vgg19 = models.vgg19(pretrained=True).to(device)

    try:
        net.load_state_dict(torch.load('./MobileNetV2_200.mod'))
        #net = torch.load('./lambda=1.pth')
    except Exception:
        print('no such file')
    else:
        print('Successfully load net model')

    # --------------before kmeans--------------
    test(net)


    # --------------sort second_grad------------
    second_grad = torch.load('./second_grad.pt')
    second_grad_descending, indices = torch.sort(torch.abs(second_grad), 0, descending=True)

    parm = net.parameters()
    for i, p in enumerate(parm):

        if i == 0:
            weights = p.data.view(-1, 1)
        else:
            weights = torch.cat((weights, p.data.view(-1, 1))).to(device)

    l = weights.size()[0]
    order_weights = torch.zeros(l, 1).to(device).scatter_(0, indices, weights.clone())
    print(order_weights.size())
    #print(weights)
    #print(order_weights)

    degree = []
    i = 100
    p = 0
    while (p < second_grad.size()[0]):
        degree.append(i)
        p = p + i
        i = i * 10

        if p + i > second_grad.size()[0]:
            i = second_grad.size()[0] - p

    degree.append(second_grad.size()[0])

    print(degree)
    # print(second_grad_descending)
    # spilte_indices = torch.split(indices, degree)

    spilte_weights = torch.split(order_weights, degree)
    print('success')
    #
    # for idx, splite_param in enumerate(spilte_weights):
    #     if idx == 0:
    #         after_kmean_param = splite_param
    #
    #     else:
    #         cluster_ids_x, cluster_centers = kmeans(
    #             X=splite_param, num_clusters=256, distance='euclidean', device=device, tol=0.0005
    #         )
    #         predict_label = kmeans_predict(splite_param.reshape(-1, 1), cluster_centers, 'euclidean', device=device)
    #         netweight_quanti = cluster_centers[predict_label].type(torch.cuda.FloatTensor)
    #         after_kmean_param = torch.cat([after_kmean_param, netweight_quanti])
    #         print(netweight_quanti.data.mean())

    # print(order_weights)
    # print(after_kmean_param)



    def original_order(ordered, indices):
        z = torch.empty_like(ordered)
        for i in range(ordered.size(0)):
            z[indices[i]] = ordered[i]
        return z


    # unsorted = original_order(after_kmean_param, indices)
    unsorted = original_order(order_weights, indices)


    parm_ = net.parameters()

    each_layers_param_num = []
    for param in parm_:
        #print(name, type(param))
        each_layers_param_num.append(param.data.view(-1, 1).size()[0])

    print(each_layers_param_num)
    each_layers_param = torch.split(unsorted, each_layers_param_num)

    param_with_name = net.named_parameters()
    for idx, param in enumerate(each_layers_param):
        for idx_x, (name_id, parm) in enumerate(param_with_name):
            if idx == idx_x:
                net_change = operator.attrgetter(name_id)(net)
                net_change.data.copy_(nn.parameter.Parameter(param.reshape(parm.data.size()).type(torch.cuda.FloatTensor)))

    #print(after_degree)

    # ------------------after kmeans---------------------
    test(net)
