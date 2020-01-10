import numpy as np
import random
import h5py

def extract_random(full_imgs, full_msks, patch_h, patch_w, N_patches, inside=True): 
    if (N_patches % full_imgs.shape[0] != 0):
        print("N_patches: please enter a mutiple of 20")
    assert (len(full_imgs.shape)==4 and len(full_msks.shape)==4)
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3) #check channel is 1 or 3
    assert (full_msks.shape[1]==1) #check msk channel is 1
    assert (full_imgs.shape[2] == full_msks.shape[2] and full_imgs.shape[3] == full_msks.shape[3])
    patches = np.empty((N_patches, full_imgs.shape[1], patch_h, patch_w))
    patches_msk = np.empty((N_patches, full_msks.shape[1], patch_h, patch_w))
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]

    patch_per_img = int(N_patches/full_imgs.shape[0]) #equally divide the full image
    print("patch per full image: " + str(patch_per_img))
    iter_tot = 0
    for i in range(full_imgs.shape[0]):
        #record patches number
        k  = 0
        while k < patch_per_img:
            x_center = random.randint(0+int(patch_w/2), img_w-int(patch_w/2))
            y_center = random.randint(0+int(patch_h/2), img_h-int(patch_h/2))
            if inside==True:
                if is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h)==False:
                    continue
            
            #get patches
            patch = full_imgs[i, :, y_center-int(patch_h/2):y_center+int(patch_h/2), x_center-int(patch_w/2):x_center+int(patch_w/2)]
            
            patch_msk = full_msks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2), x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot] = patch
            patches_msk[iter_tot] = patch_msk
            iter_tot += 1
            k += 1
    return patches, patches_msk



def is_patch_inside_FOV(x, y, img_w, img_h, patch_h):
    x = x - int(img_w/2)
    y = y - int(img_h/2)
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0)
    radius = np.sqrt((x*x) + (y*y))
    if radius < R_inside:
        return True
    else:
        return False


def consistency_check(imgs, msks):
    '''
    assert(imgs.shape[0] == msks.shape[0])
    assert(imgs.shape[2] == msks.shape[2])
    assert(imgs.shape[3] == msks.shape[3])
    assert(msks.shape[1] == 1)
    '''
    assert(imgs.shape[1] == 1 or imgs.shape[1] == 3)
    assert(np.max(msks)==1 and np.min(msks)==0), 'max: %.4f, min: %.4f' %(np.max(msks), np.min(msks))


def load_hdf5(infile):
    with h5py.File(infile, 'r') as f:
        return f['image'][()]


def mask_transform(msk):
    assert(len(msk.shape) == 4)
    n_patches = msk.shape[0]
    msk_h = msk.shape[2]
    msk_w = msk.shape[3]
    msk = np.reshape(msk, (n_patches, msk.shape[1], msk_h*msk_w))
    new_msk = np.empty((n_patches, msk_h*msk_w, 2))
    assert(np.max(msk) == 1 and np.min(msk)==0)
    for i in range(msk.shape[0]):
        for j in range(msk.shape[2]):
            if msk[i,:,j] == 0:
                new_msk[i,j,0] = 1
                new_msk[i,j,1] = 0
            else:
                new_msk[i,j,0] = 0
                new_msk[i,j,1] = 1
    return new_msk



















