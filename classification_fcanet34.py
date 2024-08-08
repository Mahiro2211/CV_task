import torchmetrics.classification
import torch, torchvision
import torch.nn as nn
from torchvision import models
import numpy as np
import pandas as pd
import scipy
from torch import optim
import argparse
from models import Load_Model_CLS
from load_data import *
import torchmetrics
import random
from loguru import logger
import os
import datetime

parser = argparse.ArgumentParser(
                    prog='Classification BaseLine Param',
                    description='data,models and optimizer setting',
                    epilog='Text at the bottom of help')

parser.add_argument("--path",default="output_img",help="image path")
parser.add_argument("--eval",default=1)
parser.add_argument("--epoch",default=50,help="training epoch")
parser.add_argument("--device",default='cuda:0',help="device")
parser.add_argument("--pretrained",default=False)
parser.add_argument("--seed",default=2004)


parser.add_argument("--num_class",default=4,help="dataset class")
parser.add_argument("--loss",default="CrossEntropyLoss",help='loss in torch.nn')
parser.add_argument("--lr",default=1e-4,help="learning rate")
parser.add_argument("--opt",default="SGD")
parser.add_argument("--bs",default=16,help="Batch size")
parser.add_argument("--RNN",default="False",help="Batch size")

class Engine():

    def __init__(self) -> None:
        self.arg = parser.parse_args()
        self.device = torch.device(self.arg.device)
        self.model = Load_Model_CLS(num_class=self.arg.num_class,RNN=self.arg.RNN).to(self.device)
        self.criterion = getattr(nn,self.arg.loss)()
        self.load_opt = getattr(optim,str(self.arg.opt))
        self.opt = self.load_opt(self.model.parameters(),self.arg.lr)
        self.load_data()
        self.train_loss_list = np.zeros((self.arg.epoch))
        self.test_loss_list = np.zeros((self.arg.epoch))



        self.eval_acc = torchmetrics.functional.classification.accuracy
        
    def load_data(self):
        self.train_dataset, self.test_dataset, self.train_loader, self.test_loader = \
            getImgDatasetDataLoader(self.arg.path,self.arg.bs)
    
    def train_(self,epoch): # train_one epoch
        
        self.model.train()
        all_loss = 0
        for data, label in self.train_loader:
            data = data.to(self.device)
            label = label.to(self.device)
            
            pred = self.model(data)
            loss = self.criterion(pred,label)

            self.opt.zero_grad()
            loss.backward()
            all_loss += loss
            self.opt.step()
        logger.info(f'Training Epoch {epoch}: Loss:{all_loss/len(self.train_dataset)}')
        return all_loss / len(self.train_dataset)
    
    def test_(self,epoch):
        self.model.eval()
        total_acc = []
        for data, label in self.test_loader:
            data = data.to(self.device)

            label = label.to(self.device)
            pred = self.model(data)
            
            # 先softmax然后取最大索引，和标签对齐 
            pred = torch.nn.functional.softmax(pred)
            pred = torch.argmax(pred,dim=1)
            
            total_acc.append(self.eval_acc(pred.cpu().detach(),label.cpu().detach(),task="multiclass", num_classes=4,))
        total_acc = torch.stack(total_acc)
        logger.info(f'Epoch {epoch}:Acc:{torch.mean(total_acc)}')

if __name__ == "__main__":
    script = Engine()
    
    #################### 保证实验可复现 #####################
    torch.backends.cudnn.deterministic=True
    np.random.seed(script.arg.seed); random.seed(script.arg.seed)
    torch.manual_seed(script.arg.seed); torch.cuda.manual_seed(script.arg.seed); torch.cuda.manual_seed_all(script.arg.seed)

    s = 'log_dir'
    os.makedirs(s,exist_ok=True)
    logger.add(s + '/' + str(script.arg.lr) + str(script.arg.loss) + str(script.arg.opt) + '.log', format="{time} {level} {message}", filter="my_module", level="INFO")

    for i in range(script.arg.epoch):
        train_loss = script.train_(i + 1)
        script.train_loss_list[i] = train_loss
        if script.arg.epoch % script.arg.eval == 0:
            script.test_(i + 1)