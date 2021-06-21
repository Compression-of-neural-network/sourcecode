from MyModel import MyModel
import torch
from kmeans_pytorch import kmeans, kmeans_predict
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    net = MyModel().to(device)

    try:
        net.load_state_dict(torch.load('./MyModel.mod'))
    except Exception:
        print('no such file')
    else:
        print('Successfully load net model')

    #before kmeans
    net.Mytest()

    #kmeans
    parm = net.parameters()
    for i, p in enumerate(parm):
        if i == 0:
            weights = p.data.view(-1, 1)
        else:
            weights = torch.cat((weights, p.data.view(-1, 1)))

    cluster_ids_x, cluster_centers = kmeans(
        X=weights, num_clusters=256, distance='euclidean', device=device
    )

    conv1_quantized = kmeans_predict(net.conv1.weight.reshape(-1, 1), cluster_centers, 'euclidean', device=device)
    net.conv1.weight = nn.parameter.Parameter(
        conv1_quantized.reshape(net.conv1.weight.size()).type(torch.cuda.FloatTensor))

    conv2_quantized = kmeans_predict(net.conv2.weight.reshape(-1, 1), cluster_centers, 'euclidean', device=device)
    net.conv2.weight = nn.parameter.Parameter(
        conv2_quantized.reshape(net.conv2.weight.size()).type(torch.cuda.FloatTensor))

    fc_quantized = kmeans_predict(net.fc1.weight.reshape(-1, 1), cluster_centers, 'euclidean', device=device)
    net.fc1.weight = nn.parameter.Parameter(
        fc_quantized.reshape(net.fc1.weight.size()).type(torch.cuda.FloatTensor))

    #after kmeans
    net.Mytest()