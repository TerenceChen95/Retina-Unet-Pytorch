import os

path = os.path.split(__file__)[0]
config = {
        'n_classes' : 1,
        'batch_size' : 128,
        'num_epochs' : 800,
        'lr' : 0.001,
        'device' : 'cuda:0',
        'save_pth' : '%s/saved' % path,
        'patch_width' : 48,
        'patch_height' : 48,
        'N_imgs' : 20,
        'N_subimgs' : 19000,
        'inside_FOV' : False,
        'stride_h' : 5,
        'stride_w' : 5
        }
