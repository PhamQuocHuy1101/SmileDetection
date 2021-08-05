import torch
import torch.nn as nn
from torchvision import models

class Model(nn.Module):
    def __init__(self, name):
        super(Model, self).__init__()
        self.net = self.__create_model(name)
    def __create_model(self, name):
        if name == 'lenet':
            model = models.googlenet(pretrained = True)
        elif name == 'resnet18':
            model = models.resnet18(pretrained = True)
        elif name == 'resnet50':
            model = models.resnet50(pretrained = True)
        else:
            return None
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2) #2 class
        return model
    def forward(self, X):
        # restnet18_au_best padding 0
        # padding = torch.zeros_like(X)
        # padding_X = torch.cat([X, padding, padding], dim = 1)

        padding_X = torch.repeat_interleave(X, 3, dim = 1)
        return self.net(padding_X)