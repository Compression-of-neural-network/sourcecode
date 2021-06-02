import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    net = torch.load('./mnist_cnn.pth')
    parm = net.parameters()
    weights = parm.__next__().view(-1)
    for i, n in enumerate(parm):
        if i == 0:
            continue
        else:
            weights = torch.cat([weights, n.view(-1)])

    kmeans_cluster = KMeans(n_clusters=256, max_iter=10)
    kmeans_cluster.fit(weights.to('cpu').data.numpy().reshape(-1, 1))

