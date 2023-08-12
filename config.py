import cv2
import torch
import os
from math import log2
import torch
import pathlib

START_TRAIN_AT_IMG_SIZE = 8
DATASET = 'paiva' # Nome do dataset
CHECKPOINT_GEN = "generator.pth"  # Nome do checkpoint
CHECKPOINT_CRITIC = "critic.pth" # Nome do checkpoint
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu" # Nome do Device
TENSORBOARD = True
PROFILING = False # Profiling
SAVE_MODEL = True # Salvar modelo
LOAD_MODEL = False # Carregar modelo
WHERE_LOAD = "PAS_07-07-2023-18h06m57s_ADAMW" # Onde carregar o modelo
GENERATE_IMAGES = True # Gerar imagens
N_TO_GENERATE = 5 # Gerar quantas imagens
VIDEO = False # Fazer vídeo (nao implementado ainda)
GENERATED_EPOCH_DISTANCE = 50 # Gerar imagem a cada quantas epochs
LEARNING_RATE_GENERATOR = 1e-3
LEARNING_RATE_DISCRIMINATOR = 1e-3
WEIGHT_DECAY = 0 #0.001
MIN_LEARNING_RATE = 5e-7
PATIENCE_DECAY = 10 # Caso o scheduler esteja True, a cada x epochs ele será atualizado/reiniciado
BATCH_SIZES = [64, 64, 64, 64, 16, 8, 2, 1, 1]
IMAGE_SIZE = 256 # Tamanho da imagem de saida
CHANNELS_IMG = 3 # Numero de canais da imagem
SIMULATED_STEP = 8 # Quantidade de passos para gerar imagem. 2^(simulated_step + 1) = tamanho da imagem naquele momento
Z_DIM = 256 # Tamanho do espaco latente do modelo
W_DIM = 256 # Tamanho do espaço latente para os estilos
IN_CHANNELS = 256  # Tamanho do input do modelo
LAMBDA_GP = 20 # Valor para o multiplicador do Gradient Penalty
NUM_STEPS = int(log2(IMAGE_SIZE/4)) 
PROGRESSIVE_EPOCHS = [20*10,40*10,80*10,80*10,160*10,640*10,640*10] * len(BATCH_SIZES) # Epochs para cada tamanho de imagem 
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = int(os.cpu_count()/4) #Numero de threads para carregar as imagens no dataset
DIFF_AUGMENTATION = False
OPTMIZER = "ADAMW" # ADAM / NADAM / RMSPROP / ADAMAX / ADAMW / ADAMW8 / ADAGRAD / SGD / SWA / SAM / RADAM
SCHEDULER = True # Utilizar LearningRate Scheduler
SCHEDULER_MULT = 1 # Multplier que utilizamos para separar cada "iteracao" do scheduler. ! Somente Ints !
MODEL = "Style" # Style, Pro, Wavelet
CHANNEL_LAST = False
CREATE_MODEL_GRAPH = False
SPECIAL_NUMBER = 1e-5 # Evitar divisao por zero
CRITIC_TREAINING_STEPS = 1 # A cada quantos steps treinar o critic
ACCUM_ITERATIONS = 1 # Gradient acumulation - Quantas epochs acumular por gradient antes de fazer o backpropagation
FOLDER_PATH = str(pathlib.Path().resolve())
FID = False  # Calcular o FID
PAUSE_LAYERS = True