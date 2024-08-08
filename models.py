import torch, torchvision
import torch.nn as nn
from torchvision import models
import numpy as np
import pandas as pd
import scipy

class Load_Model_CLS(nn.Module):
    def __init__(self,num_class:int, *args, **kwargs) -> None:
        super(Load_Model_CLS,self).__init__()
        # self.backbone = getattr(models,model)(True)
        self.backbone = torch.hub.load('cfzd/FcaNet', 'fca34' ,pretrained=False)
        if kwargs['RNN'] == False :
            self.backbone.fc = nn.Linear(512,num_class,bias=True)
        elif kwargs['RNN'] == 'GRU':
            self.backbone.fc = RNN(input_size, hidden_size, num_layers, num_classes)
        else:
            pass

    def forward(self,x):
        return self.backbone(x)

