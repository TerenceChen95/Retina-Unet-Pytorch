import torch.nn as nn
import torch
import numpy as np
from config import config
import torch.nn.functional as F

class UNET(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNET, self).__init__()
        self.conv0 = conv3x3(in_channels, 32, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
       
        self.bn0 = nn.BatchNorm2d(32)
        #set all 3x3 filter's padding equals to 1 so that
        #the output shape doesn't change
        self.conv1 = conv3x3(32, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool_0 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = conv3x3(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = conv3x3(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = conv3x3(64, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = conv3x3(128, 128, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = conv3x3(128, 64, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = conv3x3(64, 64, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = conv3x3(64, 32, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(32)
        self.conv9 = conv3x3(32, 32, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(32)
        
        self.conv10 = conv1x1(32, n_classes, 1, 1)
        self.bn10 = nn.BatchNorm2d(1)

        self.upconv0 = upconv2x2(128, 64, 2, 2)
        self.bn_up1 = nn.BatchNorm2d(64)
        self.upconv1 = upconv2x2(64, 32, 2, 2)
        self.bn_up2 = nn.BatchNorm2d(32)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        #x = nn.Dropout2d(p=0.2)(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #stage1 feature map
        stage1 = x
        x = self.maxpool_0(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #x = nn.Dropout2d(p=0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        #stage2 feature map
        stage2 = x
        x = self.maxpool_0(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        #x = nn.Dropout2d(p=0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.upconv0(x)
        x = self.bn_up1(x)
        x = torch.cat([x, stage2], dim=1)
        x = self.conv6(x) 
        x = self.bn6(x)
        x = self.relu(x)
        #x = nn.Dropout2d(p=0.2)(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        
        x = self.upconv1(x)
        x = self.bn_up2(x)
        x = torch.cat([x, stage1], dim=1)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        #x = nn.Dropout2d(p=0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        #x = x.view(x.shape[0],x.shape[1], x.shape[2]*x.shape[3])
        #x = x.permute(0, 2, 1)
        #x = F.softmax(x) 
        #x = x.reshape((x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
        #x = x.permute(0,2,1)
        #x = self.activate(x)

        return x



# W2 = (W1-K+2P)/S+1
def conv3x3(in_c, out_c, k, s, p):
    return nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)

#W2 = (W1-1)*S-2*P+K
def upconv2x2(in_c, out_c, k, s):
     return nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s)


def conv1x1(in_c, out_c, k, s):
    return nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s)


def concat(c1, c2):
    return torch.cat([c1,c2], dim=1)


def cut(c1, c2):
    x1,y1 = c1.size()[2:]
    x2,y2 = c2.size()[2:]
    c2 = c2[:, :, int((x2-x1)/2) : int((x1+x2)/2), int((y2-y1)/2) : int((y2+y1)/2)]
    return c2
