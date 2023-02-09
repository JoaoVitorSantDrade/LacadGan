import cv2
import torch
import os
from math import log2

START_TRAIN_AT_IMG_SIZE = 4
DATASET = 'glomerulo_normal'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = True
GENERATE_IMAGES = True
N_TO_GENERATE = 5
GENERATED_EPOCH_DISTANCE = 5
LEARNING_RATE_GENERATOR = 8e-5 #0.00008
LEARNING_RATE_DISCRIMINATOR = 9e-5 #0.00009
BATCH_SIZES = [128, 64, 64, 32, 32, 16, 8, 4, 2] # 4 8 16 32 64 128 256 512 1024
IMAGE_SIZE = 512
CHANNELS_IMG = 3
Z_DIM = 512 # 512 no paper
IN_CHANNELS = 512  # 512 no paper
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE/4))
PROGRESSIVE_EPOCHS = [40, 45, 55, 55, 60, 60, 65, 65, 70] * len(BATCH_SIZES) # Pra cada tamanho de imagem | 30 -> 4H numa GT 1030
FIXED_NOISE = torch.randn(4, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 0 #int(os.cpu_count()) Demora mto para carregar e o dataset n é grande pra valer a pena
DIFF_AUGMENTATION = True
RESTART_LEARNING = False

# 30 Progressive Epochs
# 256 Z_DIM e IN_CHANNELS
# 4x4 -> 4h numa GT 1030

# 10 Progressive Epochs
# 128 Z_DIM e IN_CHANNELS
# 64x64 -> 8h 40m numa GT 1030