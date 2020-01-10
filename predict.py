import torch
from PIL import Image
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from models.net2 import UNET
from torchvision import transforms as transforms
import torch.nn.functional as F
from config import config
from posprocess import rgb2gray, pad_border, recover_overlap, get_data_testing_overlap, clahe_equal, adjust_gamma
from torch.utils.data import DataLoader
import cv2

def prepro(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[9:574,:]
    img = clahe_equal(img)
    img = adjust_gamma(img)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    return img

img = '/home/tianshu/unet/data/training/images/22_training.tif'
state = torch.load('./saved/BEST_checkpoint.pth.tar', map_location={'cuda:1':'cuda:0'})['model']
net = UNET(1, 1)
net.load_state_dict(state)
device = torch.device('cuda:0')

#normalize input
img = prepro(img)
stride_h = config['stride_h']
stride_w = config['stride_w']
patch_h = config['patch_height']
patch_w = config['patch_width']
patches_img_test, new_h, new_w = get_data_testing_overlap(img, patch_h, patch_w, stride_h, stride_w)
batch_size = config['batch_size']
#batch_size = 32
test_loader = DataLoader(patches_img_test, batch_size=batch_size, shuffle=False)

net = net.to(device)
net.eval()
outsize = patches_img_test.shape

msks = np.empty((outsize))
activate = torch.nn.Sigmoid()
#patches too large to be put into model all at once
for i, data in enumerate(test_loader):
    with torch.no_grad():
        data = data.to(device, dtype=torch.float32)
        msk = net(data)
        msk = activate(msk)
        try:
            msks[i*batch_size:(i+1)*batch_size] = msk.detach().data.cpu().numpy()
        except Exception as e:
            print(e)
            msks[i*batch_size:] = msk.detach().data.cpu().numpy()
    
pred_img = recover_overlap(msks, new_h, new_w, stride_h, stride_w)
print(pred_img.shape)
pred_img = pred_img[0][0]
print(np.max(pred_img), np.min(pred_img))


threshold = np.zeros((pred_img.shape))
for j in range(pred_img.shape[1]):
    for i in range(pred_img.shape[0]):
        if pred_img[i,j] > 0.5:
            threshold[i,j] = 1
        else:
            threshold[i,j] = 0


out_img = Image.fromarray((threshold*255))
plt.figure()
plt.imshow(out_img)
plt.savefig('out.jpg')

