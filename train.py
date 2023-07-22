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
from model import Discriminator, Generator, StyleDiscriminator, StyleGenerator
from math import log2
from tqdm import tqdm
import config
import warnings
from torchcontrib.optim import SWA

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
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=32,
        persistent_workers=True,
        multiprocessing_context='spawn',
    )
    return loader, dataset

def train_fn(
    disc,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_disc,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_disc,
    scheduler_gen,
    scheduler_disc,
    now,
    ):

    all_step = len(loader)    
    loop = tqdm(loader, leave=True, unit_scale=True, smoothing=1.0, colour="cyan", ncols=80, desc="batch training")
    loss_gen = 0
    for batch_idx, (real, _) in enumerate(loop):
        # imagens por segundo batch_size * it

        # Zera os gradientes
        opt_disc.zero_grad(set_to_none = True)
        opt_gen.zero_grad(set_to_none = True) 

        cur_batch_size = 0
        real = real.cuda() #Passa as imagens p/ Cuda
        # Train Disc
        with torch.cuda.amp.autocast():
            real = real.to(memory_format=torch.channels_last) # Converte as imagens para Channel Last 
            # https://docs.fast.ai/callback.channelslast.html 
            # https://www.intel.com/content/www/us/en/developer/articles/technical/pytorch-vision-models-with-channels-last-on-cpu.html 
            # https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout

            cur_batch_size = real.shape[0]

            noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1,device=torch.device('cuda'))
            noise = noise.to(memory_format=torch.channels_last)

            if config.STYLE:
                fake = gen(noise,alpha,step)
            else:
                fake = gen(noise) # Gera imagens falsas
            
            if config.DIFF_AUGMENTATION:
                fake = DfAg.DiffAugment(fake)
                real = DfAg.DiffAugment(real)

            if config.STYLE:
                disc_real = disc(real,alpha,step) #True Positives/Negatives
                disc_fake = disc(fake.detach(),alpha,step) #False Positives/Negatives
            else:
                disc_real = disc(real) #True Positives/Negatives
                disc_fake = disc(fake.detach()) #False Positives/Negatives
            
            if config.STYLE:
                gp = gradient_penalty(disc, real, fake, alpha, step) # Calcula gradient Penalty
            else:
                gp = gradient_penalty(disc, real, fake) # Calcula gradient Penalty
            #https://paperswithcode.com/method/wgan-gp#:~:text=weight%20clipping%20parameter%20.-,A%20Gradient%20Penalty%20is%20a%20soft%20version%20of%20the%20Lipschitz,used%20as%20the%20gradient%20penalty.
            
            loss_disc = (
                -(torch.mean(disc_real) - torch.mean(disc_fake))
                + config.LAMBDA_GP * gp
                + (0.01 * torch.mean(disc_real ** 2)) 
            )

        scaler_disc.scale(loss_disc).backward()
        scaler_disc.step(opt_disc)
        scaler_disc.update()

        if batch_idx % config.CRITIC_TREAINING_STEPS == 0:
            # Train Gen
            if config.STYLE:
                gen_fake = disc(fake,alpha,step)
            else:
                gen_fake = disc(fake)
            loss_gen = -torch.nanmean(gen_fake)
            
            
            scaler_gen.scale(loss_gen).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()
        
        alpha += cur_batch_size / ((len(dataset) * config.PROGRESSIVE_EPOCHS[step]*0.20))
        alpha = min(alpha, 1)

        if not config.STYLE:
            gen.set_alpha(alpha)
            disc.set_alpha(alpha)

        if batch_idx  % 200 == 0 and config.TENSORBOARD:    
            with torch.no_grad(): # Plotar no tensorboard
                if config.STYLE:
                    fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
                else:
                    fixed_fakes = gen(config.FIXED_NOISE) * 0.5 + 0.5
                plot_thread = Thread(target=plot_to_tensorboard, args=(writer, loss_disc.detach(), loss_gen.detach(), real.detach(), fixed_fakes.detach(), tensorboard_step, now, gen, disc, gp,), daemon=True)
                plot_thread.start()
            tensorboard_step += 1

    if config.FID:
        FID_Score = calculate_fid(real,fake).item()
            
    if config.SCHEDULER:
        scheduler_gen.step()
        scheduler_disc.step()


    if config.OPTMIZER == "SWA":
        opt_disc.swap_swa_sgd()
        opt_gen.swap_swa_sgd()

    
    
    print(f"Gradient Penalty: {gp.detach()}\tAlpha: {alpha}\nScheduler: {scheduler_gen.get_lr()[0]}\tFID: {FID_Score}")
    return tensorboard_step, alpha

def main():
    now = datetime.now()
    
    print(f"Versão do PyTorch: {torch.__version__}\nGPU utilizada: {torch.cuda.get_device_name(torch.cuda.current_device())}\nDataset: {config.DATASET}\nData-Horario: {now.strftime('%d/%m/%Y - %H:%M:%S')}")
    print(f"CuDNN: {torch.backends.cudnn.version()}\n")
    tensorboard_step = 0

    if config.STYLE:
        gen = StyleGenerator(config.Z_DIM, config.W_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE, non_blocking=True, memory_format=torch.channels_last)
        disc = StyleDiscriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE, non_blocking=True, memory_format=torch.channels_last)
    else:
        gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE, non_blocking=True, memory_format=torch.channels_last)
        disc = Discriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE, non_blocking=True, memory_format=torch.channels_last)
        if config.CREATE_MODEL_GRAPH:
            plot_cnns_tensorboard()
            config.CREATE_MODEL_GRAPH = False

    #Initialize optmizer and scalers for FP16 Training
    match config.OPTMIZER:
        case "RMSPROP":
            opt_gen = optim.RMSprop(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, weight_decay=config.WEIGHT_DECAY, foreach=True,momentum=0.5)
            opt_disc = optim.RMSprop(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, weight_decay=config.WEIGHT_DECAY, foreach=True, momentum=0.5)
        case "ADAM":
            opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER,weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER,weight_decay=config.WEIGHT_DECAY)
        case "NADAM":
            opt_gen = optim.NAdam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.NAdam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "ADAMAX":
            opt_gen = optim.Adamax(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adamax(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "ADAMW": 
            # https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
            # https://openreview.net/forum?id=ryQu7f-RZ&noteId=B1PDZUChG
            opt_gen = optim.AdamW(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.AdamW(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "ADAGRAD":
            opt_gen = optim.Adagrad(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR,eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adagrad(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR,eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
        case "SGD":
            opt_gen = optim.SGD(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR,momentum=0.9, nesterov=True, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.SGD(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR,momentum=0.9, nesterov=True, weight_decay=config.WEIGHT_DECAY)
        case "SAM": #https://github.com/davda54/sam https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam
            #opt_gen = optim.AdamW
            #opt_disc = optim.AdamW
            #opt_gen = sam.SAM(gen.parameters(),opt_gen, lr=config.LEARNING_RATE_GENERATOR, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
            #opt_disc = sam.SAM(disc.parameters(),opt_gen, lr=config.LEARNING_RATE_GENERATOR, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
            raise NotImplementedError("Sam is not implemented")
        case "SWA":
            baseGen = optim.AdamW(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.99),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            baseDisc = optim.AdamW(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_gen = SWA(baseGen,swa_start=10,swa_freq=5,swa_lr=0.05)
            opt_disc = SWA(baseDisc,swa_start=10,swa_freq=5,swa_lr=0.05)
        case "ADAMW8":
            if os.name == 'nt':
                raise NotImplementedError("ADAMW8 do not work on Windows")
            baseGen = bnb.optim.AdamW8bit(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
            baseDisc = bnb.optim.AdamW8bit(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)        
        case "RADAM":
            opt_gen = optim.RAdam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.RAdam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.9,0.999),eps=config.SPECIAL_NUMBER, weight_decay=config.WEIGHT_DECAY)
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
        
        for i in range(step, step_s[0]):
            tensorboard_step += config.PROGRESSIVE_EPOCHS[i]

        step = step_s[0]
        cur_epoch = epoch_s[0] + 1
        tensorboard_step += cur_epoch
        schedulerGen.last_epoch, schedulerDisc.last_epoch = cur_epoch


    images_saw = 0
    

    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4*2**step)
        img_size = 4*2**step
        
        if not config.STYLE:
            gen.set_alpha(alpha)
            gen.set_step(step)
            disc.set_alpha(alpha)
            disc.set_step(step)
        for epoch in range(cur_epoch,num_epochs):
            gen.train()
            disc.train()
            print(f"Epoch [{epoch}/{num_epochs}] - Tamanho:{img_size} - Batch size:{config.BATCH_SIZES[step]}")
            tensorboard_step, alpha = train_fn(
                disc,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_disc,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_disc,
                schedulerGen,
                schedulerDisc,
                now,
            )


            images_saw = images_saw + config.BATCH_SIZES[step] * len(loader)
            print(f"Images saw: {images_saw}")

            if config.LOAD_MODEL:
                data_save_path = config.WHERE_LOAD
            else:
                data_save_path = config.DATASET + "_"+ now.strftime("%d-%m-%Y-%Hh%Mm%Ss") + "_" + config.OPTMIZER

            if config.GENERATE_IMAGES and (epoch%config.GENERATED_EPOCH_DISTANCE == 0) or epoch == (num_epochs-1) and config.GENERATE_IMAGES:
                img_generator = Thread(target=generate_examples, args=( gen, step, config.N_TO_GENERATE, (epoch-1), (4*2**step), data_save_path,), daemon=True)
                try:
                    img_generator.start()
                    img_generator.join()
                except Exception as err:
                    print(f"Erro: {err}")

            if config.SAVE_MODEL:
                gen_check = Thread(target=save_checkpoint, args=(gen, opt_gen, schedulerGen, epoch, step, config.CHECKPOINT_GEN, data_save_path,), daemon=True)
                critic_check = Thread(target=save_checkpoint, args=(disc, opt_disc, schedulerDisc, epoch, step, config.CHECKPOINT_CRITIC, data_save_path,), daemon=True)
                try:
                    gen_check.start()
                    critic_check.start()
                    gen_check.join()
                    critic_check.join()
                except Exception as err:
                    print(f"Erro: {err}")

        "requires_grad = false em todas camadas que já treinou"
        for module_name, module in gen.named_modules():
            if f"prog_blocks.{step}" in module_name:
                module.require_grad = False

        step += 1
        cur_epoch = 0


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.emit_nvtx = False
    torch.set_num_threads(4)

    with keepawake(keep_screen_awake=False):
        warnings.filterwarnings("ignore")
        path = config.FOLDER_PATH
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', path, '--bind_all'])
        url = tb.launch()
        print(f"\n\nTensorboard rodando em {url}")
        main()
