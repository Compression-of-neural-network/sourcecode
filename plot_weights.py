from matplotlib import pyplot as plt
import torch
import numpy


def plot_weights(model):
    modules = [module for module in model.modules()]
    num_sub_plot = 0
    for i, layer in enumerate(modules):
        if hasattr(layer, 'weight'):
            plt.subplot(131 + num_sub_plot)
            w = layer.weight.data
            w_one_dim = w.cpu().numpy().flatten()
            plt.hist(w_one_dim[w_one_dim != 0], bins=50)
            num_sub_plot += 1
    plt.show()


def plot_all_weights(model):
    modules = [module for module in model.modules()]
    all_weights = torch.tensor([])
    for i, layer in enumerate(modules):
        if hasattr(layer, 'weight'):
            w = layer.weight.data.view(-1).cpu()
            all_weights = torch.cat((all_weights, w))
    plt.plot()
    plt.hist(all_weights.cpu().numpy(), bins=50)
    plt.show()
