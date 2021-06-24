from Models import *
import torch.nn.utils.prune as prune
import operator
import plot_weights


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# test_data = MNIST('./data', train=False, download=True, transform=transformer)
# test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_data = CIFAR10('./data', train=False, download=True, transform=transformer)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()


def test(net, f):
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
                                                                              accuracy), file=f)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset),
                                                                              accuracy))




if __name__ == '__main__':
    #net = MyModel_MNIST().to(device)
    net = VGG16().to(device)
    #net = MobileNetV2().to(device)
    pruning_rate = 0.99
    log_file = open("./pruning_log/pruning_log_VGG16_{}.log".format(pruning_rate), "w")
    # -------------load model-----------------
    try:
        net.load_state_dict(torch.load('./saved_models_after_training/VGG16_200.mod'))
    except Exception:
        print('no such file')
        exit()
    else:
        print('Successfully load net model')
    # --------------Before pruning--------------
    print('Before pruning:', file=log_file)
    print('Before pruning:')
    test(net, log_file)
    plot_weights.plot_all_weights(net)
    # ---------------pruning----------------------
    # parameters_to_prune = (
    #     (net.conv1, 'weight'),
    #     (net.conv2, 'weight'),
    #     (net.fc1, 'weight'),
    # )
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=pruning_rate,
    # )

    parameters_to_prune = ()
    print(type(parameters_to_prune))
    parameters = net.named_parameters()
    for name, param in parameters:
        print(name.split("."))
        name = name.split(".")
        if name[1] == 'weight':
            x = operator.attrgetter(name[0])(net)
            x = ((x, 'weight'), (x, 'bias'))
            parameters_to_prune = parameters_to_prune + x


    print(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )

    print('Pruning rate: {}'.format(pruning_rate), file=log_file)
    print('Pruning rate: {}'.format(pruning_rate))
    # ------------------After pruning---------------------
    print('After pruning:', file=log_file)
    print('After pruning:')
    test(net, log_file)
    plot_weights.plot_all_weights(net)
    log_file.close()
