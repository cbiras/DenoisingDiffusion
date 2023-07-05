# This is a sample Python script.
import math

import torch

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from dataset import *
from helper import *
import matplotlib.pyplot as plt

class SimpleDiffusion:
    def __init__(self,
                 num_timestemps=1000,
                 device='cpu',
                 img_shape=(3,32,32)
                 ):
        self.num_timestemps = num_timestemps
        self.device = device
        self.img_shape = img_shape
        self.intialize()

    def initialize(self):
        self.betas = self.get_betas()
        self.alpha = 1 - self.betas
        self.alpha_cum = torch.cumprod(self.alpha,dim=0)
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cum)
        self.one_over_sqrt_alpha = 1./torch.sqrt(self.alpha)
        self.one_minus_sqrt_alpha_cum = torch.sqrt(1 - self.alpha_cum)

    def get_betas(self):
        scale = 1000/self.num_timestemps
        start = scale * 1e-4
        end = scale * 0.02
        return torch.linspace(
            start=start,
            end=end,
            steps=self.num_timestemps,
            device=self.device,
            dtype=torch.float32
        )

def forward_difussion(sd: SimpleDiffusion, x: torch.Tensor, timestamp: torch.Tensor):
    noise = torch.rand_like(x) #noise
    mean = get(sd.sqrt_alpha_cum,timestamp)
    std = get(sd.one_minus_sqrt_alpha_cum,timestamp)
    sample = mean * x + std * noise

    return std * noise
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    loader = get_dataloader(dataset_name=BaseConfig.DATASET,batchsize=128)
    loader = iter(loader)
    batch,_ = next(loader)

    noisy_images = []
    specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]

    plt.figure(figsize=(24,12),facecolor='white')

    for b_image, _ in loader:
        b_image = inverse_tranform(b_image).cpu()
        grid_img = make_grid(b_image/255.0,nrow=16,padding=True,pad_value=1,normalize=True)
        plt.imshow(grid_img.permute(1,2,0))
        plt.axis('off')
        plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
