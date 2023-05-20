import cv2
import torch
import os
from math import log2
import torch
import pathlib

START_TRAIN_AT_IMG_SIZE = 16
DATASET = 'test_aug'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
PROFILING = False
SAVE_MODEL = True
LOAD_MODEL = False
WHERE_LOAD = "test_aug_aug_19_04_07h_02m"
GENERATE_IMAGES = False
N_TO_GENERATE = 5
VIDEO = False
GENERATED_EPOCH_DISTANCE = 1
LEARNING_RATE_GENERATOR = 3e-4 #0.005
LEARNING_RATE_DISCRIMINATOR = 3e-4 #0.005
WEIGHT_DECAY = 0 #0.001
MIN_LEARNING_RATE = 9e-5
PATIENCE_DECAY = 15
BATCH_SIZES = [128, 128, 64, 64, 16, 8, 8, 8, 8] # 4 8 16 32 64 128 256 512 1024
IMAGE_SIZE = 256
CHANNELS_IMG = 3
SIMULATED_STEP = 7
Z_DIM = 256 # 512 no paper
IN_CHANNELS = 256  # 512 no paper
LAMBDA_GP = 20
LAMBDA_LX = 0.01
NUM_STEPS = int(log2(IMAGE_SIZE/4))
PROGRESSIVE_EPOCHS = [60] * len(BATCH_SIZES) # Pra cada tamanho de imagem | 30 -> 4H numa GT 1030
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = int(os.cpu_count()/4) #Demora mto para carregar e o dataset n é grande pra valer a pena
DIFF_AUGMENTATION = False
RESTART_LEARNING = False
RESTART_LEARNING_TIMEOUT = 20
OPTMIZER = "ADAMW" # ADAM / NADAM / RMSPROP / ADAMAX / ADAMW / ADAMW8 / ADAGRAD / SGD / SWA / SAM / RADAM
SCHEDULER = False
STYLE = False
CREATE_MODEL_GRAPH = False
SPECIAL_NUMBER = 1e-5
CRITIC_TREAINING_STEPS = 1
ACCUM_ITERATIONS = 5
FOLDER_PATH = str(pathlib.Path().resolve())

# 30 Progressive Epochs
# 256 Z_DIM e IN_CHANNELS
# 4x4 -> 4h numa GT 1030

# 10 Progressive Epochs
# 128 Z_DIM e IN_CHANNELS
# 64x64 -> 8h 40m numa GT 1030