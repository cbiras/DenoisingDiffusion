from dataclasses import dataclass
from helper import get_default_device
import os
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
    NUM_WORKERS = 0

@dataclass
class ModelConfig:
    BASE_CH = 64
    BASE_CH_MUL = (1,2,4,4)
    APPLY_ATTETION = (False,True,True,False)
    DROPOUT_RATE=0.1
    TIME_EMB_MUL=4
