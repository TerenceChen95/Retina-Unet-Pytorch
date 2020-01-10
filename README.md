# Retina-Unet-Pytorch
A Pytorch implement of retina vessel segementation

## Dataset
DRIVE dataset is provided in ./data directory

## Networks
I define 3 types of unet model in the ./models directory
- original_unet: implemented as the essay describe
- net2: add padding for convolution/deconvolution kernels in order to maintain input shape
- net_improve: add batch_normalization for each layer to converge faster

## Some results
![](./results/out.jpg)

![](./results/out_net_improve.jpg)
