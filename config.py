import cv2
import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 8
DATASET = 'chest_xray'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
GENERATE_IMAGES = True
N_TO_GENERATE = 5
GENERATED_EPOCH_DISTANCE = 10
LEARNING_RATE_GENERATOR = 7e-4 #0.0007
LEARNING_RATE_DISCRIMINATOR = 9e-4 #0.0009
BATCH_SIZES = [64, 64, 32, 32, 16, 16, 8, 4, 2]
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 64  # 512 no paper
IN_CHANNELS = 64  # 512 no paper
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE/4)) + 1
PROGRESSIVE_EPOCHS = [50] * len(BATCH_SIZES) # Pra cada tamanho de imagem | 30 -> 4H numa GT 1030
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4

# 30 Progressive Epochs
# 256 Z_DIM e IN_CHANNELS
# 4x4 -> 4h numa GT 1030

# 10 Progressive Epochs
# 128 Z_DIM e IN_CHANNELS
# 64x64 -> 8h 40m numa GT 1030