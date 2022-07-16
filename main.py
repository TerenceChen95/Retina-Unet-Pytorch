import os
from torch.utils.data import DataLoader
from config import config
from solver import Solver
from dataset import Dataset
from torchvision import transforms
from utils import load_hdf5, consistency_check, extract_random, mask_transform
import numpy as np
import torch

def get_data_training(img_hdf5, msk_hdf5, patch_h, patch_w, N_subimgs, inside_FOV):
    train_img = load_hdf5(img_hdf5)
    train_msk = load_hdf5(msk_hdf5)
    consistency_check(train_img, train_msk)

    patches_train_img, patches_train_msk = extract_random(train_img, train_msk, patch_h, patch_w, N_subimgs, inside=inside_FOV)
    consistency_check(patches_train_img, patches_train_msk)
    #patches_train_msk = mask_transform(patches_train_msk)
    return patches_train_img, patches_train_msk


class train_dataset(object):
    def __init__(self, img_hdf5, msk_hdf5):
        patches_train_img, patches_train_msk = get_data_training(img_hdf5, msk_hdf5, config['patch_height'], config['patch_width'], config['N_subimgs'], config['inside_FOV'])
        assert(len(patches_train_img.shape) == 4)
        self.input = patches_train_img
        self.output = patches_train_msk

    def __getitem__(self, index):
        return self.input[index], self.output[index]

    def __len__(self):
        return self.input.shape[0]

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
    
    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def main():
    root = os.getcwd()+'/data/training/'
    #get training patches
    img_hdf5 = './hdf5/DRIVE_dataset_imgs_train.hdf5'
    msk_hdf5 = './hdf5/DRIVE_dataset_msks_train.hdf5'
    train_data = train_dataset(img_hdf5, msk_hdf5)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    prefetcher = data_prefetcher(train_loader)
    solver = Solver(config)
    solver.train(prefetcher, resume=False, best=False)

if __name__ == '__main__':
    main() 
