from dataset import Dataset
import os
import h5py
import numpy as np
from PIL import Image
from config import config
from torchvision import transforms

def write_hdf5(arr, outfile):
    with h5py.File(outfile, 'w') as f :
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def get_dataset(root, mode, transform=None):
    dataSet = Dataset(root, transform=transform, mode=mode)
    imgs = np.empty((config['N_imgs'], 1, 565, 565))
    msks = np.empty((config['N_imgs'], 1, 565, 565))
    #msks = np.empty((config['N_imgs'], 565*565, 2))
    for i in range(len(dataSet)):
        if mode == 'train':
            img, label = dataSet[i]
            #print(img.shape, label.shape)
            assert(img.shape == (1, 565, 565))
            #assert(label.shape == (565*565, 2))
            imgs[i] = img
            msks[i] = label
        elif mode == 'test':
            img = dataSet[i]
            assert(img.shape == (1, 565, 565))
            imgs[i] = img
    if mode == 'train':
        write_hdf5(imgs, './hdf5/DRIVE_dataset_imgs_%s.hdf5' % (mode))
        write_hdf5(msks, './hdf5/DRIVE_dataset_msks_%s.hdf5' % (mode))
    elif mode == 'test':
        write_hdf5(imgs, './hdf5/DRIVE_dataset_imgs_%s.hdf5' % (mode))


#transform = transforms.Compose([transforms.ToTensor()])
train_root = os.getcwd()+'/data/training/'
test_root = os.getcwd()+'/data/testing/'

get_dataset(train_root, mode='train')
get_dataset(test_root, mode='test')


