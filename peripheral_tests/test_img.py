from dataset import Dataset
import os
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from scipy.misc import imsave

root = os.getcwd()+'/data/training/'
img_hdf5 = './hdf5/DRIVE_dataset_imgs_train.hdf5'
msk_hdf5 = './hdf5/DRIVE_dataset_msks_train.hdf5'
transform = transforms.Compose([transforms.ToTensor()])

train_data = Dataset(root, transform, mode='test')

img = train_data[1]
img = img.squeeze(0)
print(img.shape)
img = np.array(img)
imsave('input.jpg', img)
