import numpy as np
import torch
import cv2

def recover_overlap(preds, img_h, img_w, stride_h, stride_w):
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h + 1
    N_patches_w = (img_w-patch_w)//stride_w + 1
    N_patches_img = N_patches_h * N_patches_w

    N_full_img = preds.shape[0] // N_patches_img
    probs = np.zeros((N_full_img, preds.shape[1], img_h, img_w))
    overlaps = np.zeros((N_full_img, preds.shape[1], img_h, img_w))

    k = 0
    for i in range(N_full_img):
        for m in range(N_patches_h):
            for n in range(N_patches_w):
                probs[i,:,m*stride_h:m*stride_h+patch_h, n*stride_w:n*stride_w+patch_w]+=preds[k]
                overlaps[i,:,m*stride_h:m*stride_h+patch_h, n*stride_w:n*stride_w+patch_w]+=1
                k += 1
    average = probs/overlaps
    return average

                
def pad_border(full_imgs, patch_h, patch_w, stride_h, stride_w):
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    leftover_h = (img_h-patch_h) % stride_h
    leftover_w = (img_w-patch_w) % stride_w
    if (leftover_h != 0):
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_h+(stride_h-leftover_h),img_w))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_h,0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    
    if (leftover_h != 0):
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],full_imgs.shape[2],img_w+(stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:full_imgs.shape[2],0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    '''
    if (leftover_h != 0) and (leftover_w != 0):
        print('Padding horizontal and vertical.')
        tmp_full_img = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_h+leftover_h, img_w+leftover_w))
        tmp_full_img[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_h, 0:img_w] = full_imgs
    '''
    return full_imgs


def rgb2gray(img):
    gray = img[:,:,0]*0.299+img[:,:,1]*0.587+img[:,:,2]*0.114
    gray = np.reshape(gray, (img.shape[0], img.shape[1], 1))
    return gray

def extract_ordered_overlap(full_img, patch_h, patch_w, stride_h, stride_w):
    img_h = full_img.shape[2]
    img_w = full_img.shape[3]
    N_patches_h = (img_h-patch_h) // stride_h + 1
    N_patches_w = (img_w-patch_w) // stride_w + 1
    N_patches_img = N_patches_h*N_patches_w
    N_patches_tot = N_patches_img*full_img.shape[0]
    patches = np.empty((N_patches_tot, full_img.shape[1], patch_h, patch_w))
    iter_tot = 0
    for i in range(full_img.shape[0]):
        for m in range(N_patches_h):
            for n in range(N_patches_w):
                patch = full_img[i,:,m*stride_h:m*stride_h+patch_h, n*stride_w:n*stride_w+patch_w]
                patches[iter_tot] = patch
                iter_tot += 1
    assert(iter_tot == N_patches_tot)
    return patches



def get_data_testing_overlap(img, patch_h, patch_w, stride_h, stride_w):
    img = np.asarray(img)
    test_img = pad_border(img, patch_h, patch_w, stride_h, stride_w)
    patches_img_test = extract_ordered_overlap(test_img, patch_h, patch_w, stride_h, stride_w) 
    patches_img_test = torch.from_numpy(patches_img_test)
    return patches_img_test, test_img.shape[2], test_img.shape[3]

def clahe_equal(img):
        img_equalized = np.empty(img.shape)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_equalized[:,:] = clahe.apply(np.array(img[:,:], dtype=np.uint8))
        return img_equalized
    
def adjust_gamma(img, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_img = np.empty(img.shape)
    new_img[:,:] = cv2.LUT(np.array(img[:,:], dtype = np.uint8), table)
    return new_img

def kill_border(pred_img, msk):
    msk = msk[9:574, :]
    msk /= 255
    for y in range(pred_img.shape[1]):
        for x in range(pred_img.shape[0]):
            if msk[x,y] == 0:
                pred_img[x,y] = 0
    return msk
