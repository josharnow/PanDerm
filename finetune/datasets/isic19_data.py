# import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms.functional as F
import torch
class ISIC19Dataset(Dataset):
    def __init__(self, df: pd.DataFrame,  transforms=None):
        """
		Class initialization
		Args:
			df (pd.DataFrame): DataFrame with data description
			train (bool): flag of whether a training dataset is being initialized or testing one
			transforms: image transformation method to be applied
			meta_features (list): list of features with meta information, such as sex and age

		"""
        self.df = df
        self.transforms = transforms

        self.class_counts = dict(self.df['label'].value_counts())
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[cls]
                                for cls in self.df['label']]


    def __getitem__(self, index):
        filename = self.df.iloc[index]['image']

        im_path = '/home/share/Eval_Data/ISIC_2019_Training/'+filename+'.jpg'
        # Use PIL to load the image directly in RGB format
        try:
            x = Image.open(im_path).convert('RGB')
        except IOError:
            print('Error opening file:', im_path)
            x = None  # Or handle the error as appropriate for your application

        # Apply transformations if any
        if x is not None and self.transforms:
            x = self.transforms(x)
            y = self.df.iloc[index]['label']
            # x=x.to(torch.float64)
        # Handle whether it's training mode or not
        # if self.train:
        #     y = self.df.iloc[index]['label']
        #     return x, y
        # else:
        #     return x
        return x,y

    def __len__(self):
        return len(self.df)
