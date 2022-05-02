import sys
import os
from torch.nn import init
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as function
from torchvision.models import densenet
from collections import OrderedDict

"""
Modified from https://github.com/yaodongyu/Rethink-BiasVariance-Tradeoff/blob/master/mnist/models/nn_mnist.py
"""

class MLP1Hidden(nn.Module):
    def __init__(self, config):
        super(MLP1Hidden, self).__init__()
        if config["w_softmax"] == "yes":
            self.w_softmax = True
        else:
            self.w_softmax = False
        self.fc1 = nn.Linear(int(config["in_dim"]), int(config["width"]), bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(config["width"]), int(config["out_dim"]), bias=False)
    
    def forward(self, x):
        y = self.fc1(torch.flatten(x, 1))
        y = self.relu(y)
        y = self.fc2(y)
        if self.w_softmax:
            y = function.softmax(y, dim=1)
        return y


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

class MLLeakModel(nn.Module):
    def __init__(self, config):
        super(MLLeakModel, self).__init__()
        if config["w_softmax"] == "yes":
            self.w_softmax = True
        else:
            self.w_softmax = False

        num_classes = int(config["out_dim"])
        # compute_padding_for_same(kernel_size) replaces the setup of padding="same"
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.Tanh(),
            nn.Linear(128, num_classes)
        )
        self.net.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        if self.w_softmax:
            x = function.softmax(x, dim=1)
        return x

"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Modified from https://github.com/yaodongyu/Rethink-BiasVariance-Tradeoff/blob/master/cifar10/models/resnet.py
"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = function.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = function.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = function.relu(self.bn1(self.conv1(x)))
        out = function.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = function.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, k=1, num_classes=10, w_softmax="no"):
        super(ResNet, self).__init__()

        if w_softmax == "yes":
            self.w_softmax = True
        else:
            self.w_softmax = False

        self.in_planes = 1 * k

        self.conv1 = nn.Conv2d(3, 1 * k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1 * k)
        self.layer1 = self._make_layer(block, 1 * k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * k, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * k, num_blocks[3], stride=2)
        self.linear = nn.Linear(8 * k * block.expansion, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = function.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = function.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.w_softmax:
            out = function.softmax(out, dim=1)
        return out

class AlexNet(nn.Module):
    def __init__(self, k = 64, num_classes=100, kernel_size_conv = [4,3,3,3,3], kernel_size_pool = 2, input_size = 32, w_softmax="no"):
        super(AlexNet, self).__init__()

        if w_softmax == "yes":
            self.w_softmax = True
        else:
            self.w_softmax = False

        self.net = nn.Sequential(
            nn.Conv2d(3, k, kernel_size=kernel_size_conv[0], stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel_size_pool),
            nn.Conv2d(k, 3*k, kernel_size=kernel_size_conv[1], padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel_size_pool),
            nn.Conv2d(3*k, 6*k, kernel_size=kernel_size_conv[2], padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6*k, 4*k, kernel_size=kernel_size_conv[3], padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*k, 4*k, kernel_size=kernel_size_conv[4], padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel_size_pool),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(4* k * int(input_size/16)**2, 4096),  # (input_size/16)**2 = 2*2 is due to stride and maxpooling
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        if self.w_softmax:
            x = function.softmax(x, dim=1)
        return x

class AlexNetWOL(nn.Module):
    def __init__(self, k = 64, num_classes=100, w_softmax="no"):
        super(AlexNetWOL, self).__init__()

        if w_softmax == "yes":
            self.w_softmax = True
        else:
            self.w_softmax = False

        self.net = nn.Sequential(
            nn.Conv2d(3, k, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(k, 3*k, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3*k, 6*k, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6*k, 4*k, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*k, 4*k, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(4*k, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        if self.w_softmax:
            x = function.softmax(x, dim=1)
        return x


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False, w_softmax="no"):

        super(DenseNet, self).__init__()

        if w_softmax == "yes":
            self.w_softmax = True
        else:
            self.w_softmax = False

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = densenet._DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = densenet._Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = function.relu(features, inplace=True)
        out = function.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        if self.w_softmax:
            out = function.softmax(out, dim=1)
        return out

def alexnet(config):
    width=int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return AlexNet(k = width, num_classes=num_classes, w_softmax=w_softmax)

def alexnet_wol(config):
    width=int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return AlexNetWOL(k = width, num_classes=num_classes, w_softmax=w_softmax)


def ResNet18(config):
    width=int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return ResNet(BasicBlock, [2, 2, 2, 2], k=width, num_classes=num_classes, w_softmax=w_softmax)


def ResNet34(config):
    width=int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return ResNet(BasicBlock, [3, 4, 6, 3], k=width, num_classes=num_classes, w_softmax=w_softmax)


def ResNet50(config):
    width=int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return ResNet(BasicBlock, [3, 4, 14, 3], k=width, num_classes=num_classes, w_softmax=w_softmax)


def ResNet26_bottle(config):
    width=int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return ResNet(Bottleneck, [2, 2, 2, 2], k=width, num_classes=num_classes, w_softmax=w_softmax)


def ResNet38_bottle(config):
    width=int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return ResNet(Bottleneck, [3, 3, 3, 3], k=width, num_classes=num_classes, w_softmax=w_softmax)


def ResNet50_bottle(config):
    width=int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return ResNet(Bottleneck, [3, 4, 6, 3], k=width, num_classes=num_classes, w_softmax=w_softmax)

def densenet121(config):
    growth_rate = int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return DenseNet(
        growth_rate=growth_rate, 
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        num_classes=num_classes,
        w_softmax=w_softmax
    )

def densenet161(config):
    growth_rate = int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return DenseNet(
        growth_rate=growth_rate, 
        block_config=(6, 12, 36, 24),
        num_init_features=96,
        num_classes=num_classes,
        w_softmax=w_softmax
    )

def densenet169(config):
    growth_rate = int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return DenseNet(
        growth_rate=growth_rate, 
        block_config=(6, 12, 32, 32),
        num_init_features=64,
        num_classes=num_classes,
        w_softmax=w_softmax
    )

def densenet201(config):
    growth_rate = int(config["width"])
    num_classes = int(config["out_dim"])
    w_softmax = config["w_softmax"]
    return DenseNet(
        growth_rate=growth_rate, 
        block_config=(6, 12, 48, 32),
        num_init_features=64,
        num_classes=num_classes,
        w_softmax=w_softmax
    )

MODELS = {
    "mlp_1hidden": MLP1Hidden,
    "resnet34": ResNet34,
    "densenet121": densenet121,
    "densenet161":densenet161,
    "alexnet":alexnet,
    # TODO Shiqi's code has alexnet_wol as the tag in the config so we may want to modify the
    # configs for the lower weight decay models
    "alexnetwol": alexnet_wol,
    "mlleak": MLLeakModel
}
