from dataset import Dataset
import os
from PIL import Image
import numpy as np
from utils import load_hdf5
'''
pth = '/home/tianshu/unet/data/comb/label/'
#imgs = sorted(os.listdir(pth))
img_pth = pth+'21_manual1.gif'
label = Image.open(img_pth).convert('L')
label = np.array(label)[9:574, :]
print(label.shape)
label = label / 255
classes = np.unique(label)
print(classes)
'''

msks = load_hdf5('./hdf5/DRIVE_dataset_imgs_train.hdf5')
print(np.max(msks), np.min(msks))







