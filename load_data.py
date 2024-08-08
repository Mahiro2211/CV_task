from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import ImageFolder
import PIL.Image as PII
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


def getImgDatasetDataLoader(img_path,batch_size,if_dataset=False,if_ml=False):
    
    flatten_transform = v2.Lambda(lambda x : x.reshape(-1))
    to_numpy_transform = v2.Lambda(lambda x : x.numpy())

    seq_transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((256,256)),
    v2.CenterCrop((224,224)),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    if if_ml :
        seq_transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((256,256)),
        v2.CenterCrop((224,224)),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        flatten_transform,
        to_numpy_transform,
    ])
    all_label = []

    dataset = ImageFolder(img_path,transform=seq_transform)
    
    # Collect All label to compute num of the sample so as to avoid long-tail distribution
    for _,label in dataset:
        all_label.append(label)
    
    all_label = np.array(all_label)
    train_indice, test_indice = train_test_split(range(len(all_label)), test_size=0.4,
                                                 stratify=all_label,random_state=42)
    # print(train_indice,test_indice)
    train_dataset = Subset(dataset,train_indice)
    test_dataset = Subset(dataset,test_indice)
    if if_dataset :
        return train_dataset, test_dataset
    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,num_workers=4,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,num_workers=4,pin_memory=True)

    return train_dataset, test_dataset, train_loader, test_loader

    
    



