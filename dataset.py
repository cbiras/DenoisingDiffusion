from dataclasses import dataclass
import os
import torchvision.transforms as tf
from helper import get_default_device, DeviceDataLoader
import torchvision.datasets as datasets
from torch.utils.data import Dataset,DataLoader

@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "Flowers"

    root_log_dir = os.path.join("Logs_Checkpoints", "Inference")
    root_checkpoint_dir = os.path.join("Logs_Checkpoints", "checkpoints")

    log_dir = "version_0"
    checkpoint_dir = "version_0"

@dataclass
class TrainingConfig:
    TIMESTEPS = 1000
    IMG_SHAPE = (1,32,32) if BaseConfig.DATASET == 'Mnist' else (3,32,32)
    NUM_EPOCHS = 800
    BATCH_SIZE = 32
    LR = 2e-4
    NUM_WORKERS = 2

def get_dataset(dataset_name='Flowers'):
    transforms = tf.Compose(
        [
        tf.ToTensor(),
        tf.Resize((32,32),interpolation=tf.InterpolationMode.BICUBIC,
                  antialias=True),
        tf.RandomHorizontalFlip(),
        tf.Lambda(lambda t: (t*2)-1)
        ]
    )
    if dataset_name.upper() == 'MNIST':
        dataset = datasets.MNIST(root='./data',train=True,transform=transforms,download=True)
    elif dataset_name.upper() == 'FLOWERS':
        dataset = datasets.ImageFolder(root='./data/flowers',transform=transforms)

    return dataset
def get_dataloader(dataset_name='FLOWERS',
                   batchsize=32,
                   pin_memory=False,
                   shuffle=True,
                   num_workers=0,
                   device="cpu"):

    dataset = get_dataset(dataset_name)
    dataloader = DataLoader(dataset=dataset,batch_size=batchsize,shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)

    device_dataloader = DeviceDataLoader(dataloader,device)
    return device_dataloader

def inverse_tranform(tensors):
    return ((tensors.clamp(-1,1)+1.0)/2.0) * 255.0