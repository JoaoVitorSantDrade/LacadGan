import cv2
import torch
import os
from math import log2

START_TRAIN_AT_IMG_SIZE = 4
DATASET = 'test_aug'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = True
GENERATE_IMAGES = False
N_TO_GENERATE = 5
GENERATED_EPOCH_DISTANCE = 1
LEARNING_RATE_GENERATOR = 5e-3 #0.0001
LEARNING_RATE_DISCRIMINATOR = 5e-3 #0.0001
WEIGHT_DECAY = 3e-4 #0.0003
MIN_LEARNING_RATE = 5e-5
PATIENCE_DECAY = 10
BATCH_SIZES = [64, 32, 32, 32, 16, 4, 2, 1, 1] # 4 8 16 32 64 128 256 512 1024
IMAGE_SIZE = 256
CHANNELS_IMG = 3
SIMULATED_STEP = 7
Z_DIM = 512 # 512 no paper
IN_CHANNELS = 512  # 512 no paper
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE/4))
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES) # Pra cada tamanho de imagem | 30 -> 4H numa GT 1030
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = int(os.cpu_count()/2) #Demora mto para carregar e o dataset n é grande pra valer a pena
DIFF_AUGMENTATION = False
RESTART_LEARNING = False
RESTART_LEARNING_TIMEOUT = 20
OPTMIZER = "ADAMW" # ADAM / NADAM / RMSPROP / ADAMAX / ADAMW
SCHEDULER = True
STYLE = False
CREATE_MODEL_GRAPH = True
SPECIAL_NUMBER = 1e-5

# 30 Progressive Epochs
# 256 Z_DIM e IN_CHANNELS
# 4x4 -> 4h numa GT 1030

# 10 Progressive Epochs
# 128 Z_DIM e IN_CHANNELS
# 64x64 -> 8h 40m numa GT 1030