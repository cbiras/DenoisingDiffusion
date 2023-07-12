import os.path

import torch
from torchvision.utils import make_grid
from PIL import Image
import cv2
import  base64
from IPython.display import display, HTML, clear_output

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader:
    def __init__(self , dl, device):
        self.dl = dl
        self.device=device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def get_default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def save_images(images,path,**kwargs):
    grid = make_grid(images,**kwargs)
    ndarr = grid.permute(1,2,0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get(element: torch.Tensor, t: torch.Tensor):
    e = element.gather(-1,t)
    return e.reshape(-1,1,1,1)

def setup_log_directory(config):
    if os.path.isdir(config.root_log_dir):
        folder_numbers = [int(folder.replace("version_","")) for folder in os.listdir((config.root_log_dir))]

        latest_version_number = max(folder_numbers)

        version_name = f"version_{latest_version_number +1}"
    else:
        version_name = config.log_dir

    log_dir = os.path.join(config.root_log_dir,version_name)
    checkpoint_dir = os.path.join(config.root_checkpoint_dir, version_name)

    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(checkpoint_dir,exist_ok=True)

    print(f"Logging at: {log_dir}")
    print(f"Saving checkpoint at: {checkpoint_dir}")

    return log_dir,checkpoint_dir

def frames2vid(images,save_path):
    w = images[0].shape[1]
    h = images[0].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 25, (w, h))

    for image in images:
        video.write(image)

    video.release()
    return

def display_gif(gif_path):
    b64 = base64.b64encode(open(gif_path, 'rb').read()).decode('ascii')
    display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))