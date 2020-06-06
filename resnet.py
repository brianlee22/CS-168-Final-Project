from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def initialize_resnet50(num_classes=2, feature_extract=True, use_pretrained=True):
    model_ft = models.resnet50(pretrained=use_pretrained)  # initialize resnet 50
    set_parameter_requires_grad(model_ft, feature_extract)  # set require_grad to False
    num_ftrs = model_ft.fc.in_features  # get number of features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)  # update features and number of classes
    input_size = 224

    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

