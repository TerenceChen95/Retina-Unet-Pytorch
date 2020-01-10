import os
from PIL import Image
import torch
import numpy as np
import cv2

class Dataset(object):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.img_pths = sorted(os.listdir(root+'images'))
        if self.mode == 'train':
            self.label_pths = sorted(os.listdir(root+'1st_manual'))
        self.transform = transform

    def __getitem__(self, index):
        img = self.img_pths[index]
        #img = Image.open(os.path.join(self.root+'images', img))
        if self.mode == 'train': 
            label = self.label_pths[index]
            label = Image.open(os.path.join(self.root+'1st_manual', label))
        img = cv2.imread(os.path.join(self.root+'images', img))
        '''
        img = np.array(img)
        print(img.shape)
        img = self.rgb2gray(img)
        print(img.shape)
        '''
        img = self.truncate_img(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self.clahe_equal(img)
        img = self.adjust_gamma(img)
        #img = img / 255.
        img = np.expand_dims(img, 0)
        #ToTensor: normalize to (0,1)
        '''
        if self.transform is not None:
            img = self.transform(img)
        '''
        if self.mode == 'train':
            label = np.asarray(label)
            label = label / 255
            label = label[9:574, :]
            label = np.expand_dims(label, 0)
            #label = self.get_mask(label)
            #label = self.truncate_img(label)
            #label = label.transpose(2, 0, 1)
            #label = np.reshape(label, (565*565, 2))
            label = torch.from_numpy(label).type(torch.float32)
            return img, label
        elif self.mode == 'test':
            return img

    def __len__(self):
        return len(self.img_pths)

    def rgb2gray(self, img):
        gray = img[:,:,0]*0.299+img[:,:,1]*0.587+img[:,:,2]*0.114
        gray = np.reshape(gray, (img.shape[0], img.shape[1]))
        return gray

    def truncate_img(self, img):
        img = img[9:574,:]
        return img
   

    def clahe_equal(self, img):
        img_equalized = np.empty(img.shape)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #img_equalized[0] = clahe.apply(np.array(img[0], dtype=np.uint8))
        img = clahe.apply(img)
        return img
    
    def adjust_gamma(self, img, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        new_img = np.empty(img.shape)
        new_img = cv2.LUT(np.array(img, dtype = np.uint8), table)
        return new_img

    def get_mask(self, label):
        h = label.shape[1]
        w = label.shape[0]
        new_mask = np.zeros((h, w, 2))
        for y in range(label.shape[1]):
            for x in range(565):
                if label[x,y] == 0:
                    new_mask[x,y,0] = 1
                elif label[x,y] == 255:
                    new_mask[x,y,1] = 1
        return new_mask


