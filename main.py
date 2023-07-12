# This is a sample Python script.
import gc

import torch

from config import *
from UNet import UNet
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from dataset import *
from helper import *
import matplotlib.pyplot as plt
from torchmetrics import MeanMetric
from tqdm import tqdm
from torch.cuda import amp
from torch.optim import AdamW
import torch.nn as nn
import torchvision.transforms as tf

class SimpleDiffusion:
    def __init__(self,
                 num_timestemps=1000,
                 device='cpu',
                 img_shape=(3,32,32)
                 ):
        self.num_timestemps = num_timestemps
        self.device = device
        self.img_shape = img_shape
        self.initialize()

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
    noise = torch.randn_like(x) #noise
    mean = get(sd.sqrt_alpha_cum,timestamp) * x
    std = get(sd.one_minus_sqrt_alpha_cum,timestamp)
    sample = mean + std * noise

    return sample, noise

def trainin_one_epoch(sd,epoch,model,loss,optimizer,loader,scaler: amp.GradScaler(),base_config=BaseConfig(),training_config=TrainingConfig()):

    loss_total = MeanMetric()
    model.train()

    with tqdm(total=len(loader)) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")
        for x,_ in loader:
            tq.update(1)
            ts = torch.randint(low=1,high=training_config.TIMESTEPS,size=(x.shape[0],),device=base_config.DEVICE)

            noised_img,noise = forward_difussion(sd,x,ts)

            with amp.autocast():
                predicted_noise = model(noised_img,ts)
                loss_batch = loss(noise,predicted_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss_batch).backward()

            scaler.step(optimizer)
            scaler.update()

            mean_loss_value = loss_batch.detach().item()
            loss_total.update(mean_loss_value)

            tq.set_postfix_str(s=f"Loss: {mean_loss_value:.4f}")

        mean_loss = loss_total.compute().item()

        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss
@torch.no_grad()
def sample(model, sd, timesteps=1000, img_shape=(3, 64, 64),
                      num_images=5, nrow=8, device="cpu", **kwargs):

    x = torch.randn((num_images,*img_shape),device=device)
    model.eval()

    if kwargs.get("generate_video", False):
        outs = []

    for time_step in tqdm(iterable=reversed(range(1, timesteps)),
                          total=timesteps - 1, dynamic_ncols=False,
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        pred_noise = model(x,ts)

        beta = get(sd.betas,ts)
        one_over_sqrt_alpha = get(sd.one_over_sqrt_alpha,ts)
        one_minus_sqrt_alpha_cum = get(sd.one_minus_sqrt_alpha_cum,ts)

        x = (one_over_sqrt_alpha * (x - (beta/one_minus_sqrt_alpha_cum*pred_noise)) + torch.sqrt(beta)*z)

        if kwargs.get("generate_video",False):
            x_inv = inverse_tranform(x).type(torch.uint8)
            grid = make_grid(x_inv,nrow=nrow,pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid,(1,2,0)).numpy()[:,:,::-1]
            outs.append(ndarr)

    if kwargs.get("generate_video",False):
        frames2vid(outs,kwargs['save_path'])
        display(Image.fromarray(outs[-1][:,:,::-1]))
        return None
    else:
        x = inverse_tranform(x).type(torch.uint8)
        grid = make_grid(x,nrow=nrow,pad_value=255.0).to("cpu")
        pil_image = tf.functional.to_pil_image(grid)
        pil_image.save(kwargs['save_path'],format=kwargs['save_path'][-3:].upper())
        display(pil_image)
        return None

def show_forward(sd:SimpleDiffusion,loader,**kwargs):
    iter_loader = iter(loader)
    batch, _ = next(iter_loader)

    noisy_images = []
    specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]

    for timestamp in specific_timesteps:
        timestamp = torch.as_tensor(timestamp,dtype=torch.long)

        noised,_ = forward_difussion(sd,batch,timestamp)
        noised_inversed = inverse_tranform(noised)/ 255.0
        grid_img = make_grid(noised_inversed , nrow=1, padding=1)

        noisy_images.append(grid_img)

    _, ax = plt.subplots(1,len(noisy_images),figsize=(10,5),facecolor='white')

    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.9)
    plt.axis("off")
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sd = SimpleDiffusion(TrainingConfig.TIMESTEPS,img_shape=TrainingConfig.IMG_SHAPE,device=BaseConfig.DEVICE)
    loader = get_dataloader(dataset_name=BaseConfig.DATASET,batchsize=TrainingConfig.BATCH_SIZE,device=BaseConfig.DEVICE,pin_memory=True,num_workers=TrainingConfig.NUM_WORKERS)
    model = UNet(
        input_channels=TrainingConfig.IMG_SHAPE[0],
        output_channels=TrainingConfig.IMG_SHAPE[0],
        base_channels=ModelConfig.BASE_CH,
        apply_attention=ModelConfig.APPLY_ATTETION,
        base_ch_multipliers=ModelConfig.BASE_CH_MUL,
        dropout_rate=ModelConfig.DROPOUT_RATE,
        time_multiply=ModelConfig.TIME_EMB_MUL
    )
    model.to(BaseConfig.DEVICE)

    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())
    generate_video = False
    ext = ".mp4" if generate_video else ".png"

    optimizer = AdamW(params=model.parameters(),lr=TrainingConfig.LR)
    loss=nn.MSELoss()

    scaler=amp.GradScaler()

    #show_forward(sd,loader)

    for epoch in range(1,TrainingConfig.NUM_EPOCHS+1):
        torch.cuda.empty_cache()
        gc.collect()


        trainin_one_epoch(sd,epoch, model, loss, optimizer, loader,scaler)

        if epoch % 20 == 0:
            save_path = os.path.join(log_dir, f"{epoch}{ext}")

            sample(model, sd, timesteps=TrainingConfig.TIMESTEPS, num_images=32,
                              generate_video=generate_video,
                              save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE,
                              )

            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model": model.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
            del checkpoint_dict