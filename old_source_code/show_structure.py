from torchsummary import summary
from old_source_code.MobileNetV2 import *
import sys

if __name__ == '__main__':
    log_file = open("structure.log", "w")
    original_stdout = sys.stdout
    sys.stdout = log_file

    net1 = MobileNetV2().to('cuda:0')
    #net2 = MyModel_cifar().to('cuda:0')
    net1.load_state_dict(torch.load('./MobileNetV2_200.mod'))
    #net2.load_state_dict(torch.load('MyModel_cifar.mod'))
    #net1.Mytest()
    summary(net1, (3, 32, 32))
    #summary(net2, (3, 32, 32))

    sys.stdout = original_stdout

    log_file.close()