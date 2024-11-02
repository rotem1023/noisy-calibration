import os
import torch

import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def _get_cur_file_path():
    return f'{os.path.dirname(os.path.abspath(__file__))}'

def _get_isic_dir():
    return f'{_get_cur_file_path()}/isic'
    

def get_loaders(batch_size = 32):
    norm_mean = (0.49139968, 0.48215827, 0.44653124)
    norm_std = (0.24703233, 0.24348505, 0.26158768)
    train_transform = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    isic_dir = _get_isic_dir()
        
    # Create datasets using ImageFolder
    train_dataset = datasets.ImageFolder(f'{isic_dir}/train', transform=train_transform)
    train_eval_dataset = datasets.ImageFolder(f'{isic_dir}/train', transform=val_transform)
    val_dataset = datasets.ImageFolder(f'{isic_dir}/valid', transform=val_transform)
    test_dataset = datasets.ImageFolder(f'{isic_dir}/test', transform=val_transform)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader_at_eval = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        

    return train_loader, train_loader_at_eval, val_loader, test_loader