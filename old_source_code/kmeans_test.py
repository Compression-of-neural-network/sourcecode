import torch
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans, kmeans_predict

np.random.seed(123)

data_size, dims, num_clusters = 1000, 2, 3
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', device=device
)

print(cluster_ids_x)
print(cluster_centers)

y = np.random.randn(5, dims) / 6
y = torch.from_numpy(y)

cluster_ids_y = kmeans_predict(
    y, cluster_centers, 'euclidean', device=device
)

print(cluster_ids_y)

cluster_value = cluster_centers[cluster_ids_y]
print(cluster_value)