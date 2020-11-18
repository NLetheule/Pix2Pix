# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:20:45 2020

@author: natsl
"""

import torch.nn as nn

class discriminator_block(nn.Module):
    def __init__(self,  in_channels, out_channels):
        super().__init__()
        self.discri_block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.discri_block(x)

class PatchGAN(nn.Module):
    def __init__(self, n_channels = 4):
        super(PatchGAN, self).__init__()
        
        self.inc = nn.Sequential( 
            nn.Conv2d(4, 64, kernel_size = 3, padding = 1), 
            nn.LeakyReLU(inplace=True))
        self.down1 = discriminator_block(64, 128)
        self.down2 = discriminator_block(128, 256)
        self.down3 = discriminator_block(256, 512)
        self.down4 = discriminator_block(512, 512)
        self.outc = nn.Conv2d(512, 1, kernel_size = 3, padding = 1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        logits = self.outc(x5)
        return logits