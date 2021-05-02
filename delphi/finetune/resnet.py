import torch
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import sys
import os
import numpy as np
import time
from tqdm import tqdm
from delphi.utils import AverageMeter

class ResNet(nn.Module):
    def __init__(self, params):
        super(ResNet, self).__init__()
        assert params.model_arch in ['resnet18',
                                 'resnet34',
                                 'resnet50',
                                 'resnet101',
                                 'resnet152',
                                 'resnet50-swav']
        if params.model_arch == 'resnet50-swav':
            model = torch.hub.load('facebookresearch/swav', 'resnet50')
        elif params.model_arch == 'resnet18':
            model = models.resnet18(pretrained=params.pretrained)
        elif params.model_arch == 'resnet34':
            model = models.resnet34(pretrained=params.pretrained)
        elif params.model_arch == 'resnet50':
            print("RESNET50 ARCH")
            model = models.resnet50(pretrained=True)
        elif params.model_arch == 'resnet101':
            model = models.resnet101(pretrained=params.pretrained)
        elif params.model_arch == 'resnet152':
            model = models.resnet152(pretrained=params.pretrained)
        self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        if params.feature_extractor:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.fc = nn.Linear(2048, params.num_cats)

    def forward(self, x):
        outputs = []
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        outputs.append(output)
        outputs.append(features)
        return outputs
