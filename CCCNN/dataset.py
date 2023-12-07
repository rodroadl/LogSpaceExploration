'''
dataset.py

Last edited by: GunGyeom James Kim
Last edited at: Dec 7th, 2023

Custom dataset
Transform by MaxResize - Contrast Normalization - Randomly sample 32 by 32 patches
'''

import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from util import read_16bit_png, ContrastNormalization, RandomPatches, MaxResize

class CustomDataset(Dataset):
    def __init__(self, data_dir, label_file, num_patches, log_space=False):
        '''
        constructor

        Parameters:
            data_dir(str or Path) - path for directory containing images
            lable_file(str or Path) - path for label file
            num_patches(int) - Number of patches that gets to pass into RandomPatches
            log_space(Bool, default:False, optional) - Flag whether to map chromaticity space to logarithmic space
        '''
        self.images_dir = Path(data_dir)
        self.labels = pd.read_csv(label_file)
        self.images = os.listdir(self.images_dir)
        self.num_patches = num_patches
        self.log_space = log_space

    def __getitem__(self, idx):
        '''
        Return an images and labels for given index

        Parameters:
            idx(int) - index

        Return:
            image(sequence of tensors)
            label(sequence of tensors)
        '''
        image = read_16bit_png(os.path.join(self.images_dir,self.images[idx]))
        label = torch.tensor(self.labels.iloc[idx, :].astype(float).values, dtype=torch.float32) # GehlerShi

        # transform
        black_level = 129 if self.images[idx][:3] == "IMG" else 0 # GehlerShi
        transform = transforms.Compose([
            MaxResize(1200),
            ContrastNormalization(black_level=black_level),
            RandomPatches(patch_size = 32, num_patches = self.num_patches)
        ])
        image = transform(image)

        if self.log_space: # GehlerShi: [0,1] -> [0, 4095] -> [0, ~8.3]
            image *= 4095 
            image = torch.where(image < 0, torch.log(image), 0.) # NOTE: disconinuity at 1

        return image, torch.stack([label] * image.shape[0], dim=0)
    
    def __len__(self):
        '''
        Return the length of the dataset

        Return:
            length(int)
        '''
        return len(self.images)

class ReferenceDataset(Dataset):
    def __init__(self, data_dir, label_file):
        '''
        Constructor

        Parameters:
            data_dir(str or Path) - path for directory containing images
            lable_file(str or Path) - path for label file
        '''
        self.images_dir = Path(data_dir)
        self.labels = pd.read_csv(label_file)
        self.images = os.listdir(self.images_dir)

    def __getitem__(self, idx):
        '''
        Return an image and label for given index

        Parameters:
            idx(int) - index

        Return:
            image(tensor)
            label(tensos)
        '''
        name = self.images[idx]
        image = read_16bit_png(os.path.join(self.images_dir,self.images[idx]))
        label = torch.tensor(self.labels.iloc[idx, :].astype(float).values, dtype=torch.float32) # GehlerShi
        
        return image, label, name
    
    def __len__(self):
        '''
        Return the length of the dataset

        Return:
            length(int)
        '''
        return len(self.images)
