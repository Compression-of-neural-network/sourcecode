import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    second_grad_Mobile = torch.load('second_grad.pt').cpu().numpy()
    max_value = np.amax(second_grad_Mobile)
    min_value = np.amin(second_grad_Mobile)
    fig1 = plt.figure()
    plt.title("MobileNetV2")
    plt.hist(second_grad_Mobile, bins=1000)
    plt.figtext(.7, .7, "max = %f" % max_value)
    plt.figtext(.7, .8, "min = %f" % min_value)
    #plt.show()
    fig1.savefig('MobileNet.png', dpi=1000)

    second_grad_cifar = torch.load('second_grad_cifar.pt').cpu().numpy()
    max_value = np.max(second_grad_cifar)
    min_value = np.min(second_grad_cifar)
    fig2 = plt.figure()
    plt.title("MyModel_cifar")
    plt.hist(second_grad_cifar, bins=1000)
    plt.figtext(.7, .7, "max = %f" % max_value)
    plt.figtext(.7, .8, "min = %f" % min_value)
    #plt.show()
    fig2.savefig('cifar.png', dpi=1000)

    second_grad_mnist = torch.load('second_grad_MNIST.pt').cpu().numpy()
    max_value = np.max(second_grad_mnist)
    min_value = np.min(second_grad_mnist)
    print(max_value)
    print(min_value)
    fig3 = plt.figure()
    plt.title("MyModel_cifar")
    plt.hist(second_grad_mnist, bins=1000)
    plt.figtext(.7, .7, "max = 1.237874e-15")
    plt.figtext(.7, .8, "min = -5.0388344e-16")
    #plt.show()
    fig3.savefig('MNIST.png', dpi=1000)
