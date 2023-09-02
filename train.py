import os
import torch.backends.cuda
import torch.backends.cudnn
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from wakepy import keepawake 
import diffAugmentation as DfAg
from torch.utils.data import DataLoader
from threading import Thread
from datetime import datetime
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
    plot_cnns_tensorboard,
    remove_graphs,
    calculate_fid
)
from model import Discriminator, Generator, MappingNetwork, get_noise, get_w, PathLengthPenalty, WaveDiscriminator, WaveGenerator
from math import log2
from tqdm import tqdm
import config
import warnings
from torchcontrib.optim import SWA
from torchmetrics.image.fid import FrechetInceptionDistance
from pytorch_wavelets import DWTForward, DWTInverse


mem_format = torch.channels_last if config.CHANNEL_LAST == True else torch.contiguous_format

def load_tensor(x): 
        """
        Carrega um tensor no CPU e converte todos os dados para torch.float32

        Usado em get_loader
        
        :return: o Tensor já carregado no Device
        """
        x = torch.load(x, map_location="cpu")
        x.to(torch.float32)
        return x

def get_loader(image_size):
    """
    Retorna um Loader e Um Dataset

    image_size: Dimensão das imagens que vão ser lidas do dataset
    """
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    
    dataset = datasets.DatasetFolder(root=f"Datasets/{config.DATASET}_aug/{image_size}x{image_size}",loader=load_tensor,extensions=['.pt'])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # True
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=16,
        persistent_workers=True,
        multiprocessing_context='spawn',
    )
    return loader, dataset

def train_fn(
    critic,
    gen,
    map,
    path_length_penalty,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    opt_map,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,
    scaler_map,
    scheduler_gen,
    scheduler_disc,
    now,
    fid
    ):
    # TODO: Como nao eh mais progressive epochs, devemos arrumar na Main o loop de treinamento

    loop = tqdm(loader, leave=True, unit_scale=True, smoothing=1.0, colour="cyan", ncols=200, desc="Batch training")
    loss_gen = 0
    for batch_idx, (real, _) in enumerate(loop):
        # imagens por segundo batch_size * it

        cur_batch_size = 0
        real = real.cuda() #Passa as imagens p/ Cuda

        # Train Disc
        
        real = real.to(memory_format=mem_format) # Converte as imagens para Channel Last 
            # https://docs.fast.ai/callback.channelslast.html 
            # https://www.intel.com/content/www/us/en/developer/articles/technical/pytorch-vision-models-with-channels-last-on-cpu.html 
            # https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout

        cur_batch_size = real.shape[0]

        w     = get_w(cur_batch_size, map)
        noise = get_noise(cur_batch_size)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            #https://paperswithcode.com/method/wgan-gp#:~:text=weight%20clipping%20parameter%20.-,A%20Gradient%20Penalty%20is%20a%20soft%20version%20of%20the%20Lipschitz,used%20as%20the%20gradient%20penalty.
            match config.MODEL:
                case "Style":
                    fake = gen(w, noise)
                    critic_fake = critic(fake.detach())
    
                    critic_real = critic(real)
                    gp = gradient_penalty(critic, real, fake)
                    loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake))
                        + config.LAMBDA_GP * gp
                        + (0.001 * torch.mean(critic_real ** 2))
                    )

                case "Wavelet":
                    fake = gen(w, noise)
                    critic_fake = critic(fake.detach())
    
                    critic_real = critic(real)
                    gp = gradient_penalty(critic, real, fake)
                    loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake))
                        + config.LAMBDA_GP * gp
                        + (0.001 * torch.mean(critic_real ** 2))
                    )

        critic.zero_grad(set_to_none=True)
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        gen_fake = critic(fake)
        loss_gen = -torch.mean(gen_fake)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            if batch_idx % 16 == 0:
                plp = path_length_penalty(w, fake)
                if not torch.isnan(plp):
                    loss_gen = loss_gen + plp
                    loop.set_postfix(plp=plp.detach(),refresh=False)
                    
            if config.FID:
                if batch_idx % 28 == 0:
                    FID_Score = calculate_fid(real,fake,fid)
                    if not torch.isnan(FID_Score):
                        loss_gen = loss_gen + FID_Score
                        loop.set_postfix(FID=FID_Score.detach(),refresh=False)


        map.zero_grad(set_to_none=True)
        gen.zero_grad(set_to_none=True)
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()
        opt_map.step()
        # Zera os gradientes
        

        loop.set_postfix(gp=gp.detach(), loss_critic=loss_critic.detach(),refresh=False)

        if torch.isnan(gp) or torch.isnan(loss_critic) :
            exit("Gradientes explodiram!")

        
        if batch_idx  % 20 == 0 and config.TENSORBOARD:    
            with torch.no_grad(): # Plotar no tensorboard
                match config.MODEL:
                    case "Style":
                        fixed_fakes = gen(get_w(config.FIXED_BATCH_SIZE,map),get_noise(config.FIXED_BATCH_SIZE)) * 0.5 + 0.5

                plot_thread = Thread(target=plot_to_tensorboard, args=(writer, loss_critic.detach(), loss_gen.detach(), real.detach(), fixed_fakes.detach(), tensorboard_step, now, gen, critic, gp,), daemon=True)
                plot_thread.start()
            tensorboard_step += 1

        

        
    
            
    if config.SCHEDULER:
        scheduler_gen.step()
        scheduler_disc.step()


    if config.OPTMIZER == "SWA":
        opt_critic.swap_swa_sgd()
        opt_gen.swap_swa_sgd()

    
        
    return tensorboard_step, alpha

def main():
    now = datetime.now()
    
    print(f"Versão do PyTorch: {torch.__version__}\nGPU utilizada: {torch.cuda.get_device_name(torch.cuda.current_device())}\nDataset: {config.DATASET}\nData-Horario: {now.strftime('%d/%m/%Y - %H:%M:%S')}")
    print(f"CuDNN: {torch.backends.cudnn.version()}\n")
    tensorboard_step = 0
    match config.MODEL:
        case "Style":
            gen = WaveGenerator(log_resolution=config.SIMULATED_STEP,W_DIM=config.W_DIM,n_features=32).to(device=config.DEVICE, non_blocking=True, memory_format = mem_format)
            disc = Discriminator(log_resolution=config.SIMULATED_STEP).to(device=config.DEVICE, non_blocking=True,memory_format = mem_format)
            map = MappingNetwork(config.Z_DIM,config.W_DIM).to(device=config.DEVICE, non_blocking=True,memory_format = mem_format)
            path_length_penalty = PathLengthPenalty(0.99).to(device=config.DEVICE, non_blocking=True,memory_format = mem_format)

        case "Wavelet":
            gen = WaveGenerator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(device=config.DEVICE, non_blocking=True, memory_format = mem_format)
            disc = WaveDiscriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(device=config.DEVICE, non_blocking=True,memory_format = mem_format)
          
    if config.CREATE_MODEL_GRAPH:
        w = get_w(1,map)
        plot_cnns_tensorboard(gen,disc, get_noise(1), w)

    fid = ""
    if config.FID:
        fid = FrechetInceptionDistance().to(device=config.DEVICE, non_blocking=True,memory_format = mem_format)

    #Initialize optmizer and scalers for FP16 Training
    match config.OPTMIZER:
        case "RMSPROP":
            opt_gen = optim.RMSprop(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, weight_decay=config.WEIGHT_DECAY, foreach=True,momentum=0.5)
            opt_disc = optim.RMSprop(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, weight_decay=config.WEIGHT_DECAY, foreach=True, momentum=0.5)
            opt_map = optim.RMSprop(map.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, weight_decay=config.WEIGHT_DECAY, foreach=True, momentum=0.5)
        case "ADAM":
            opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER,weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER,weight_decay=config.WEIGHT_DECAY)
            opt_map = optim.Adam(map.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER,weight_decay=config.WEIGHT_DECAY)
        case "NADAM":
            opt_gen = optim.NAdam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.NAdam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_map = optim.NAdam(map.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "ADAMAX":
            opt_gen = optim.Adamax(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adamax(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_map = optim.Adamax(map.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "ADAMW": 
            # https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
            # https://openreview.net/forum?id=ryQu7f-RZ&noteId=B1PDZUChG
            opt_gen = optim.AdamW(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.AdamW(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_map = optim.AdamW(map.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "ADAGRAD":
            opt_gen = optim.Adagrad(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR,eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adagrad(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR,eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_map = optim.Adagrad(map.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR,eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "SGD":
            opt_gen = optim.SGD(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR,momentum=0.9, nesterov=True, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.SGD(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR,momentum=0.9, nesterov=True, weight_decay=config.WEIGHT_DECAY)
            opt_map = optim.SGD(map.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR,momentum=0.9, nesterov=True, weight_decay=config.WEIGHT_DECAY)
        case "SAM": #https://github.com/davda54/sam https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam
            
            #opt_gen = sam.SAM(gen.parameters(),opt_gen, lr=config.LEARNING_RATE_GENERATOR, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
            #opt_disc = sam.SAM(disc.parameters(),opt_gen, lr=config.LEARNING_RATE_GENERATOR, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
            #opt_map = sam.SAM(map.parameters(),opt_gen, lr=config.LEARNING_RATE_GENERATOR, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
            raise NotImplementedError("Sam is not implemented")
        case "SWA":
            baseGen = optim.AdamW(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.99),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            baseDisc = optim.AdamW(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            baseMap = optim.AdamW(map.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_gen = SWA(baseGen,swa_start=10,swa_freq=5,swa_lr=0.05)
            opt_disc = SWA(baseDisc,swa_start=10,swa_freq=5,swa_lr=0.05)
            opt_map = SWA(baseMap,swa_start=10,swa_freq=5,swa_lr=0.05)
        case "ADAMW8":
            if os.name == 'nt':
                raise NotImplementedError("ADAMW8 do not work on Windows")
            else:
                baseGen = bnb.optim.AdamW8bit(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
                baseDisc = bnb.optim.AdamW8bit(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)        
        case "RADAM":
            opt_gen = optim.RAdam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.RAdam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_map = optim.RAdam(map.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case _:
            raise NotImplementedError(f"Optim function not implemented")


    # optim.lr_scheduler.CoassineWarmRestarts
    schedulerGen = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_gen,
        T_0=config.PATIENCE_DECAY, 
        T_mult=config.SCHEDULER_MULT,
        eta_min=config.MIN_LEARNING_RATE, 
        )
    schedulerDisc = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_disc,
        T_0=config.PATIENCE_DECAY, 
        T_mult=config.SCHEDULER_MULT,
        eta_min=config.MIN_LEARNING_RATE, 
        )
    

    scaler_disc = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_map = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        aux = f"{config.WHERE_LOAD}"
        aux = aux.replace(f"{config.DATASET}_","")
        writer = SummaryWriter(f"logs/LacadGan/{config.DATASET}/{aux}")
    else:
        writer = SummaryWriter(f"logs/LacadGan/{config.DATASET}/{now.strftime('%d-%m-%Y-%Hh%Mm%Ss')}-{config.OPTMIZER}")
    
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    cur_epoch = 0
    
    if config.LOAD_MODEL:
        epoch_s = [cur_epoch]
        step_s = [step]
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, epoch_s, step_s, schedulerGen, config.WHERE_LOAD
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, disc, opt_disc, epoch_s, step_s, schedulerDisc, config.WHERE_LOAD
        )
        load_checkpoint(
            config.CHECKPOINT_MAP, map, opt_map, epoch_s, step_s, dataset= config.WHERE_LOAD
        )
        for i in range(step, step_s[0]):
            tensorboard_step += config.PROGRESSIVE_EPOCHS[i]

        step = step_s[0]
        cur_epoch = epoch_s[0] + 1
        tensorboard_step += cur_epoch
        schedulerGen.last_epoch, schedulerDisc.last_epoch = cur_epoch


    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        match config.MODEL:
            case "Style":
                loader, dataset = get_loader(2**config.SIMULATED_STEP)
            case "Pro":
                loader, dataset = get_loader(4*2**step)

        img_size = 4*2**step
        
        match config.MODEL:
            case "Pro":
                gen.set_alpha(alpha)
                gen.set_step(step)
                disc.set_alpha(alpha)
                disc.set_step(step)

        for epoch in range(cur_epoch,num_epochs):
            gen.train()
            disc.train()
            map.train()
            match config.MODEL:
                case "Style":
                    print(f"Epoch [{epoch}/{num_epochs}] - Tamanho:{2**config.SIMULATED_STEP} - Batch size:{config.BATCH_SIZES[config.SIMULATED_STEP]}")
                case _:
                    print(f"Epoch [{epoch}/{num_epochs}] - Tamanho:{img_size} - Batch size:{config.BATCH_SIZES[step]}")
            tensorboard_step, alpha = train_fn(
                disc,
                gen,
                map,
                path_length_penalty,
                loader,
                dataset,
                step,
                alpha,
                opt_disc,
                opt_gen,
                opt_map,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_disc,
                scaler_map,
                schedulerGen,
                schedulerDisc,
                now,
                fid
            )

            if config.LOAD_MODEL:
                data_save_path = config.WHERE_LOAD
            else:
                data_save_path = config.DATASET + "_"+ now.strftime("%d-%m-%Y-%Hh%Mm%Ss") + "_" + config.OPTMIZER

            if config.GENERATE_IMAGES and (epoch%config.GENERATED_EPOCH_DISTANCE == 0) or epoch == (num_epochs-1) and config.GENERATE_IMAGES:
                img_generator = Thread(target=generate_examples, args=( gen, map, step, config.N_TO_GENERATE, (epoch-1), (4*2**step), data_save_path,), daemon=True)
                try:
                    img_generator.start()
                    img_generator.join()
                except Exception as err:
                    print(f"Erro: {err}")

            if config.SAVE_MODEL:
                gen_check = Thread(target=save_checkpoint, args=(gen, opt_gen, schedulerGen, epoch, step, config.CHECKPOINT_GEN, data_save_path,), daemon=True)
                critic_check = Thread(target=save_checkpoint, args=(disc, opt_disc, schedulerDisc, epoch, step, config.CHECKPOINT_CRITIC, data_save_path,), daemon=True)
                map_check = Thread(target=save_checkpoint, args=(map, opt_map, None, epoch, step, config.CHECKPOINT_MAP, data_save_path,), daemon=True)
                try:
                    gen_check.start()
                    critic_check.start()
                    map_check.start()
                    gen_check.join()
                    critic_check.join()
                    map_check.join()
                except Exception as err:
                    print(f"Erro: {err}")

        "requires_grad = false em todas camadas que já treinou"
        if config.PAUSE_LAYERS:
            for module_name, module in gen.named_modules():
                if f"prog_blocks.{step}" in module_name or f"rgb_layers.{step}" in module_name:
                    module.require_grad = False
            for module_name, module in disc.named_modules():
                if f"prog_blocks.{step}" in module_name or f"rgb_layers.{step}" in module_name:
                    module.require_grad = False
            for module_name, module in map.named_modules():
                if f"prog_blocks.{step}" in module_name or f"rgb_layers.{step}" in module_name:
                    module.require_grad = False
        step += 1
        cur_epoch = 0


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.emit_nvtx = False
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_default_dtype(torch.float32)
    with keepawake(keep_screen_awake=False):
        warnings.filterwarnings("ignore")
        path = config.FOLDER_PATH
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', path, '--bind_all'])
        url = tb.launch()
        print(f"\n\nTensorboard rodando em {url}")
        main()
