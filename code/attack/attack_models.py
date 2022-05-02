"""
These models are used in shokri_shadow_model_attack as attack models trained using the data we obtain from the shadow models
"""
import sys
from torch.nn import init
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet
from collections import OrderedDict



class SoftmaxModel(nn.Module):
    def __init__(self, n_in, n_out, w_softmax):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_out)
        self.w_softmax = w_softmax

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        y = x
        if self.w_softmax == "yes":
            y = F.softmax(x)
        return y

def softmax_net(config):
    # width=int(config["width"])
    num_classes = int(config["in_dim"])
    w_softmax = config["w_softmax"]
    net = SoftmaxModel(n_in=num_classes, n_out = 2, w_softmax = w_softmax)
    return net

class SimpleNet(nn.Module):

    def __init__(self, hidden_layer_config=(50), num_classes=100, w_softmax="no", out_dim=2):
        super(SimpleNet, self).__init__()
        if w_softmax == "yes":
            self.w_softmax = True
        else:
            self.w_softmax = False
        self.layers = [nn.Linear(num_classes, hidden_layer_config[0]), nn.ReLU(inplace=True)]
        for i in range(1,len(hidden_layer_config)):
            self.layers.append(nn.Linear(hidden_layer_config[i-1], hidden_layer_config[i]))
            self.layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*self.layers,
            nn.Linear(hidden_layer_config[-1], out_dim)
        )


    def forward(self, x):
        x = self.net(x)
        return x

def simplenet(config):
    # width=int(config["width"])
    num_classes = int(config["in_dim"])
    try:
        hidden_layer_config = list(map(int, config["hidden_layer_config"].split(",")))
    except:
        raise Exception("hidden_layer_config required in model configuration for simplenet.")
    w_softmax = config["w_softmax"]
    net = SimpleNet(hidden_layer_config = hidden_layer_config, num_classes=num_classes, w_softmax=w_softmax)
    return net


MODELS = {
    "simplenet": simplenet,
    "softmax_net": softmax_net,
}