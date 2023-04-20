import torch.backends.cuda
import torch.backends.cudnn
import torch.nn.functional as functional
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.profiler
import diffAugmentation as DfAg
from torch.utils.data import DataLoader
from threading import Thread
from wakepy import keepawake
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
    remove_graphs
)
from model import Discriminator, Generator, StyleDiscriminator, StyleGenerator
from math import log2
from tqdm import tqdm
import config
import pathlib
import warnings
from torchcontrib.optim import SWA

def load_tensor(x):
        x = torch.load(x,map_location=torch.device(config.DEVICE))
        return x


def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            #transforms.Normalize(
                #[0.5 for _ in range(config.CHANNELS_IMG)], #antes era 0.5
                #[0.5 for _ in range(config.CHANNELS_IMG)], #antes era 0.5
            #),
        ]
    )

    
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    #dataset1 = TensorDataset(torch.arange(10).view(10, 1))
    #dataset = datasets.ImageFolder(root=f"Datasets/{config.DATASET}/{image_size}x{image_size}")
    
    dataset = datasets.DatasetFolder(root=f"Datasets/{config.DATASET}/{image_size}x{image_size}",loader=load_tensor,extensions=['.pt'])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=24,
        persistent_workers= True,
        multiprocessing_context='spawn'
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
    prof
    ):

    loop = tqdm(loader, leave=True, unit_scale=True, desc="Training step")


    for batch_idx, (real, _) in enumerate(loop):
        opt_disc.zero_grad(set_to_none = True)
        opt_gen.zero_grad(set_to_none = True)

        real = real.contiguous(memory_format=torch.channels_last)
        #if config.PROFILING:
            #if batch_idx >= (0 + 1 + 3) * 2:
                #break
        
        cur_batch_size = real.shape[0]

        # Train Disc
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1,device=torch.device('cuda'))
        noise = noise.contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast():
            
            fake = gen(noise)
            
            if config.DIFF_AUGMENTATION:
                fake = DfAg.DiffAugment(fake)
                real = DfAg.DiffAugment(real)

            disc_real = disc(real) #True Positives/Negatives
            disc_fake = disc(fake.detach()) #False Positives/Negatives

            gp = gradient_penalty(disc, real, fake)
            loss_disc = (
                -(torch.nanmean(disc_real) - torch.nanmean(disc_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.nanmean(disc_real ** 2))
            )

        if config.PROFILING:
            prof.step()

        scaler_disc.scale(loss_disc).backward()
        scaler_disc.step(opt_disc)
        scaler_disc.update()

        # Train Gen
        with torch.cuda.amp.autocast():
            gen_fake = disc(fake)
            loss_gen = -torch.nanmean(gen_fake)
        
        
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()
        
        with torch.cuda.amp.autocast():
            alpha += cur_batch_size / (len(dataset) * config.PROGRESSIVE_EPOCHS[step]*0.5)
            alpha += config.SPECIAL_NUMBER
            alpha = min(alpha, 1)
            gen.set_alpha(alpha)
            disc.set_alpha(alpha)            

    with torch.no_grad():
        fixed_fakes = gen(config.FIXED_NOISE) * 0.5 + 0.5
        plot_thread = Thread(target=plot_to_tensorboard, args=(writer, loss_disc.item(), loss_gen.item(), real.detach(), fixed_fakes.detach(), tensorboard_step, now, gen, disc, gp,), daemon=True)
        plot_thread.start()           
    
    tensorboard_step += 1   
    if config.PROFILING:
        prof.stop()

    if config.OPTMIZER == "SWA":
        opt_disc.swap_swa_sgd()
        opt_gen.swap_swa_sgd()

    if config.SCHEDULER:
        scheduler_gen.step()
        scheduler_disc.step()
  
    print(f"Gradient Penalty: {gp.item()}\nAlpha: {alpha}")
    return tensorboard_step, alpha

def main():
    now = datetime.now()
    print(f"Versão do PyTorch: {torch.__version__}\nGPU utilizada: {torch.cuda.get_device_name(torch.cuda.current_device())}\nDataset: {config.DATASET}\nData-Horario: {now.strftime('%d/%m/%Y - %H:%M:%S')}")
    print(f"Profiling: {config.PROFILING}\nCuDNN: {torch.backends.cudnn.version()} ")
    tensorboard_step = 0
    prof = None
    if config.PROFILING:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./logs/LacadGan/{config.DATASET}/{now.strftime("%d-%m-%Y-%Hh%Mm%Ss")}/step_{tensorboard_step}'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        prof.start()

    if config.STYLE:
        gen = StyleGenerator(config.Z_DIM, config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
        disc = StyleDiscriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    else:
        gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
        disc = Discriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
        if config.CREATE_MODEL_GRAPH:
            plot_cnns_tensorboard()
            config.CREATE_MODEL_GRAPH = False

    #Initialize optmizer and scalers for FP16 Training
    match config.OPTMIZER:
        case "RMSPROP":
            opt_gen = optim.RMSprop(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, weight_decay=config.WEIGHT_DECAY, foreach=True,momentum=0.5)
            opt_disc = optim.RMSprop(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, weight_decay=config.WEIGHT_DECAY, foreach=True, momentum=0.5)
        case "ADAM":
            opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.99),weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99),weight_decay=config.WEIGHT_DECAY)
        case "NADAM":
            opt_gen = optim.NAdam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.NAdam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
        case "ADAMAX":
            opt_gen = optim.Adamax(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adamax(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
        case "ADAMW":
            opt_gen = optim.AdamW(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.AdamW(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
        case "ADAGRAD":
            opt_gen = optim.Adagrad(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adagrad(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, weight_decay=config.WEIGHT_DECAY)
        case "SGD":
            opt_gen = optim.SGD(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR,momentum=0.9, weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.SGD(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR,momentum=0.9, weight_decay=config.WEIGHT_DECAY)
        case "SAM": #https://github.com/davda54/sam https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam
            #opt_gen = optim.AdamW
            #opt_disc = optim.AdamW
            #opt_gen = sam.SAM(gen.parameters(),opt_gen, lr=config.LEARNING_RATE_GENERATOR, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
            #opt_disc = sam.SAM(disc.parameters(),opt_gen, lr=config.LEARNING_RATE_GENERATOR, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
            raise NotImplementedError("Sam is not implemented")
        case "SWA":
            baseGen = optim.AdamW(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
            baseDisc = optim.AdamW(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.5,0.99), weight_decay=config.WEIGHT_DECAY)
            opt_gen = SWA(baseGen,swa_start=10,swa_freq=5,swa_lr=0.05)
            opt_disc = SWA(baseDisc,swa_start=10,swa_freq=5,swa_lr=0.05) 
        case _:
            raise NotImplementedError(f"Optim function not implemented")



    #Olhar mais a fundo
    schedulerGen = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_gen,
        T_0=config.PATIENCE_DECAY-5,
        T_mult=1,
        eta_min=config.MIN_LEARNING_RATE, 
        )
    schedulerDisc = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_disc,
        T_0=config.PATIENCE_DECAY-5,
        T_mult=1,
        eta_min=config.MIN_LEARNING_RATE, 
        )

    scaler_disc = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        aux = f"{config.WHERE_LOAD}"
        aux = aux.replace(f"{config.DATASET}_","")
        writer = SummaryWriter(f"logs/LacadGan/{config.DATASET}/{aux}")
    else:
        writer = SummaryWriter(f"logs/LacadGan/{config.DATASET}/{now.strftime('%d-%m-%Y-%Hh%Mm%Ss')}")
    
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
    
    gen.train()
    disc.train()
    
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4*2**step)
        img_size = 4*2**step
        gen.set_alpha(alpha)
        gen.set_step(step)
        disc.set_alpha(alpha)
        disc.set_step(step)
        for epoch in range(cur_epoch,num_epochs):
            print(torch.cuda.memory_summary(0))
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
                prof
            )

            if config.LOAD_MODEL:
                data_save_path = config.WHERE_LOAD
            else:
                data_save_path = config.DATASET + "_"+ now.strftime("%d-%m-%Y-%Hh%Mm%Ss")

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
    
    with keepawake(keep_screen_awake=False):
        warnings.filterwarnings("ignore")
        path = str(pathlib.Path().resolve())
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', path, '--bind_all'])
        url = tb.launch()
        print(f"\n\nTensorboard rodando em {url}")
        main()
