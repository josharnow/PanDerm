# import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms.functional as F
import torch
class Sim_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, root, train=False,val=False,test=False,transforms=None,binary=False,data_percent=1):
        """
		Class initialization
		Args:
			df (pd.DataFrame): DataFrame with data description
			train (bool): flag of whether a training dataset is being initialized or testing one
			transforms: image transformation method to be applied
			meta_features (list): list of features with meta information, such as sex and age
		"""
        if train==True:
            self.df = df[df['split']=='train']
            half_rows = int(len(self.df) * data_percent)
            self.df=self.df.head(half_rows)
        elif val==True:
            self.df = df[df['split'] == 'val']
        elif test==True:
            self.df = df[df['split'] == 'test']
        self.transforms = transforms
        self.root=root
        self.binary=binary

    # import torchsnooper
    # @torchsnooper.snoop()
    def __getitem__(self, index):
        filename1 = self.df.iloc[index]['image']
        filename2 = self.df.iloc[index]['image_second']

        im_path1 = str(self.root)+str(filename1)
        im_path2 = str(self.root) + str(filename2)
        # Use PIL to load the image directly in RGB format
        try:
            x1 = Image.open(im_path1).convert('RGB')
            x2 = Image.open(im_path2).convert('RGB')
        except IOError:
            print('Error opening file:', im_path1)
            print('Error opening file:', im_path2)
            x1 = None  # Or handle the error as appropriate for your application
            x2 = None
        # Apply transformations if any
        if x1 is not None and self.transforms:
            x1 = self.transforms(x1)
            x2 = self.transforms(x2)
            if self.binary==True:
                y = self.df.iloc[index]['binary_label']
            else:
                y = self.df.iloc[index]['label']
        return x1,x2,y

    def __len__(self):
        return len(self.df)

    def count_label(self):
        train_df = self.df[self.df['split'] == 'train']
        label_counts = train_df['binary_label'].value_counts().sort_index()
        return label_counts[0], label_counts[1]
