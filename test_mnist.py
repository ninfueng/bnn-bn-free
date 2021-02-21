"""Modified from: https://github.com/itayhubara/BinaryNet.pytorch
License: Unknown
"""
import argparse
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

from binarized_modules import BinarizeLinear
from bnn_bn import YonekawaBatchNorm1d

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=3,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BinarizeLinear(784, 1024, bias=False)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = YonekawaBatchNorm1d(1024)

        self.fc2 = BinarizeLinear(1024, 1024, bias=False)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = YonekawaBatchNorm1d(1024)

        self.fc3 = BinarizeLinear(1024, 1024, bias=False)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = YonekawaBatchNorm1d(1024)

        self.fc4 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.htanh3(x)

        x = self.fc4(x)
        return x

    def test_forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = x + self.bn1.get_int_bias()
        x = self.htanh1(x)

        x = self.fc2(x)
        x = x + self.bn2.get_int_bias()
        x = self.htanh2(x)

        x = self.fc3(x)
        # Must not use YonekawaBatchNorm1d for the last layer is the floating point.
        x = self.bn3(x)
        x = self.htanh3(x)

        # The last layer is the floating-point weights.
        x = self.fc4(x)
        return x


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch%40 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

def test(dictlist):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    dictlist['test_acc'].append(correct/len(test_loader.dataset))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_bn(dictlist):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model.test_forward(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    dictlist['bn_free_test_acc'].append(correct/len(test_loader.dataset))
    print('BN Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': cpu_count(), 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net()
    if args.cuda:
        torch.cuda.set_device(0)
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dictlist = {'test_acc': [], 'bn_free_test_acc': []}
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(dictlist)
        test_bn(dictlist)

    plt.plot(dictlist['test_acc'], label='Normal BN')
    plt.plot(dictlist['bn_free_test_acc'], label='BN free')
    plt.xlabel('Epoch')
    plt.ylabel('Test Acc')
    plt.legend(loc="lower right")
    plt.show()

    plt.plot(dictlist['test_acc'], dictlist['bn_free_test_acc'])
    plt.xlabel('Test Acc')
    plt.ylabel('Test Acc BN free')
    plt.show()
