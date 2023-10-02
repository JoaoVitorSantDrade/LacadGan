import cv2
import torch
import os
from math import log2
import torch
import pathlib

START_TRAIN_AT_IMG_SIZE = 128
DATASET = 'hypertrain' # Nome do dataset
CHECKPOINT_GEN = "generator.pth"  # Nome do checkpoint
CHECKPOINT_CRITIC = "critic.pth" # Nome do checkpoint
CHECKPOINT_MAP = "map.pth" # Nome do checkpoint
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu" # Nome do Device
PRECISION_TYPE = torch.float32
TENSORBOARD = True
PROGRESS_BAR = False
SAVE_MODEL = True # Salvar modelo
LOAD_MODEL = False # Carregar modelo
WHERE_LOAD = "Folder in ./Saves" # Onde carregar o modelo
GENERATE_IMAGES = True # Gerar imagens
N_TO_GENERATE = 5 # Gerar quantas imagens
VIDEO = False # Fazer vídeo (nao implementado ainda)
GENERATED_EPOCH_DISTANCE = 2 # Gerar imagem a cada quantas epochs
LEARNING_RATE_GENERATOR = 1e-3
LEARNING_RATE_DISCRIMINATOR = 1e-3
WEIGHT_DECAY = 0.0001 #0.001
MIN_LEARNING_RATE = 5e-5
PATIENCE_DECAY = 20 # Caso o scheduler esteja True, a cada x epochs ele será atualizado/reiniciado
BATCH_SIZES = [64, 64, 64, 64, 64, 16, 16, 8, 4, 2, 2]
SIMULATED_STEP = 8 # Quantidade de passos para gerar imagem. 4*2^(simulated_step) = tamanho da imagem naquele momento
CHANNELS_IMG = 3 # Numero de canais da imagem
Z_DIM = 256 # Tamanho do espaco latente do modelo
W_DIM = 256 # Tamanho do espaço latente para os estilos
IN_CHANNELS = 256  # Tamanho do input do modelo
LAMBDA_GP = 10 # Valor para o multiplicador do Gradient Penalty
PROGRESSIVE_EPOCHS = [20*10,40*10,80*10,80*10,160*10,640*10] * len(BATCH_SIZES) # Epochs para cada tamanho de imagem
EPOCHS = 300 
FIXED_BATCH_SIZE = 8
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = int(os.cpu_count()/4) #Numero de threads para carregar as imagens no dataset
DIFF_AUGMENTATION = False
OPTMIZER = "ADAMW" # ADAM / NADAM / RMSPROP / ADAMAX / ADAMW / ADAMW8 / ADAGRAD / SGD / SWA / SAM / RADAM
SCHEDULER = True # Utilizar LearningRate Scheduler e Não pode MODEL = Style
SCHEDULER_MULT = 1 # Multplier que utilizamos para separar cada "iteracao" do scheduler. ! Somente Ints !
MODEL = "Style" # Style, Pro, Wavelet
CHANNEL_LAST = False
CREATE_MODEL_GRAPH = True
SPECIAL_NUMBER = 1e-5 # Evitar divisao por zero
CRITIC_TREAINING_STEPS = 1 # A cada quantos steps treinar o critic
ACCUM_ITERATIONS = 1 # Gradient acumulation - Quantas epochs acumular por gradient antes de fazer o backpropagation
FOLDER_PATH = str(pathlib.Path().resolve())
FID = True  # Calcular o FID
PLP = True
PSNR = True
SSIM = True
MS_SSIM = False
HISTOLOSS = True
LAMBDA_HISTO = 5
