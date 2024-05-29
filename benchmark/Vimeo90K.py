import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from model.RIFE import Model
from model.pytorch_msssim import ssim_matlab
import torchvision.transforms as transform
import lpips
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.load_model(r'/home/jason/ECCV2022-RIFE/train_log')
model.eval()
model.device()
lpips_model = lpips.LPIPS(net='alex')
path = '/home/jason/ECCV2022-RIFE/vimeo_triplet/'
f = open(path + 'tri_testlist.txt', 'r')
psnr_list = []
ssim_list = []
lpip_list = []
for i in f:
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    print(path + 'sequences/' + name + '/im1.png')
    I0 = cv2.imread(path + 'sequences/' + name + '/im1.png')
    I1 = cv2.imread(path + 'sequences/' + name + '/im2.png')
    I2 = cv2.imread(path + 'sequences/' + name + '/im3.png')
    # I1_lp = Image.open(path + 'sequences/'+name+'/im1.png')
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    mid = model.inference(I0, I2)[0]
    ssim = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255., torch.round(mid * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    mid = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
    I1 = I1 / 255.
    psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    I1 =(torch.tensor(I1.transpose(2,0,1)).unsqueeze(0).float()).cpu()
    mid = (torch.tensor(mid.transpose(2,0,1)).unsqueeze(0).float()).cpu()
    lpip = (lpips_model(I1,mid)).detach().numpy()
    lpip_list.append(lpip)
    print("Avg PSNR: {} SSIM: {} LPIP:  {}".format(np.mean(psnr_list), np.mean(ssim_list),np.mean(lpip_list)))
