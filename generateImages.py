import config
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn.functional as functional
import torch
from model import Generator
from threading import Thread
from utils import (
    generate_examples,
    load_checkpoint
)


gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
step = config.SIMULATED_STEP
data_save_path = config.WHERE_LOAD
with torch.cuda.amp.autocast():
    load_checkpoint(
                config.CHECKPOINT_GEN, gen,dataset=config.WHERE_LOAD,inference=True
            )
    generate_examples(gen, step,n=20,name=data_save_path)
