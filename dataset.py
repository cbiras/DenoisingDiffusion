from config import *
import torchvision.transforms as tf
from helper import  DeviceDataLoader
import torchvision.datasets as datasets
from torch.utils.data import Dataset,DataLoader

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

    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader

def inverse_tranform(tensors):
    return ((tensors.clamp(-1,1)+1.0)/2.0) * 255.0