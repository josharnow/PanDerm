# import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms.functional as F
import torch
import torchsnooper
# @torchsnooper.snoop()
class OOD_Dataset(Dataset):
    def __init__(self, root,df: pd.DataFrame,image_field,target_field, transforms=None,add_extension='.jpg'):
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
        self.root=root
        self.extension=add_extension
        self.image_field = image_field
        self.target_field = target_field

    def __getitem__(self, index):
        filename = self.df.iloc[index][self.image_field]

        im_path = self.root+filename+self.extension
        # Use PIL to load the image directly in RGB format
        try:
            x = Image.open(im_path).convert('RGB')
        except IOError:
            print('Error opening file:', im_path)
            x = None  # Or handle the error as appropriate for your application

        # Apply transformations if any
        if x is not None and self.transforms:
            x = self.transforms(x)
            y = self.df.iloc[index][self.target_field]

        return x,filename,y

    def __len__(self):
        return len(self.df)
