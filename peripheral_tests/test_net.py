import torch
from models.net_improve import UNET

data = torch.randn((128, 1, 48, 48))
model = UNET(1, 2)
out = model(data)
print(out.shape)
