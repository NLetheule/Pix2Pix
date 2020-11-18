# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:17:58 2020

@author: natsl
"""

import os
import gdal
import numpy as np

from skimage import io

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def get_file_in_folder(folder):
    """
        Liste récursivement le contenu des sous-répertoires
    """
    list_file = []
    for f in os.listdir(folder):
        if os.path.isdir(folder+'/'+f): # si f est un dossier
            list_file.append(get_file_in_folder(folder+'/'+f))
        else :
            list_file.append(folder+'/'+f) 
    
    return(list_file)

def select_file_name(list_file, word):
    list_selected = []
    for file in list_file:
        if type(file) is list:
            list_selected.append(select_file_name(file, word))
        elif str(file).find(word) != -1:
                list_selected.append(file)
    return list_selected

class ImgOptiqueSAR(Dataset):
    def __init__(self, img_folder, SAR_folder, patch_size=256):
        self.imgs = []
        self.SARs = []

        #This will convert the numpy array to a tensor
        conversion = T.ToTensor()
        overlap = patch_size 

        for img_index in range(0,len(img_folder)):
            print("Working on image " + str(img_index))
            #Load the tile and the corresponding SAR truth.
            img = io.imread(img_folder[img_index])/2048
            SAR_open = gdal.Open(SAR_folder[img_index])
            SAR_band = SAR_open.GetRasterBand(1)
            SAR = SAR_band.ReadAsArray()

            for i in np.arange(patch_size//2, img.shape[0] - patch_size // 2 + 1, overlap):
                for j in np.arange(patch_size//2, img.shape[1] - patch_size // 2 + 1, overlap):
                      #Crop the image and the ground truth into patch around (i,j) and save
                      #them in self.imgs and self.SARs arrays.
                      #For the image, note that we are taking the three channels (using ":")
                      #for the 3rd dimension, and we do the conversion to tensor.
                      self.imgs.append(conversion(img[i - patch_size//2:i + patch_size // 2, j - patch_size // 2:j + patch_size // 2,:]))
                      self.SARs.append(SAR[i - patch_size//2:i + patch_size // 2, j - patch_size // 2:j + patch_size // 2])
 
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        
        img = self.imgs[idx].float()
        SAR = self.SARs[idx].float()

        return img, torch.from_numpy(SAR)