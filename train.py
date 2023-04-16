import torch.backends.cuda
import torch.backends.cudnn
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
        prefetch_factor=12,
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

    loop = tqdm(loader, leave=True, unit="Epoch(s)")
    

    for batch_idx, (real, _) in enumerate(loop):
        #if config.PROFILING:
            #if batch_idx >= (0 + 1 + 3) * 2:
                #break
        cur_batch_size = real.shape[0]

        # Train Disc
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1,device=torch.device('cuda'))

        with torch.cuda.amp.autocast():
            
            fake = gen(noise)
            
            if config.DIFF_AUGMENTATION:
                fake = DfAg.DiffAugment(fake)
                real = DfAg.DiffAugment(real)

            disc_real = disc(real)
            disc_fake = disc(fake.detach())
            gp = gradient_penalty(disc, real, fake,device=config.DEVICE)
            loss_disc = (
                -(torch.nanmean(disc_real) - torch.nanmean(disc_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.nanmean(disc_real ** 2))
            )
        if config.PROFILING:
            prof.step()

        opt_disc.zero_grad()
        scaler_disc.scale(loss_disc).backward()

        scaler_disc.unscale_(opt_disc)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 0.01)

        scaler_disc.step(opt_disc)
        scaler_disc.update()

        # Train Gen
        with torch.cuda.amp.autocast():
            gen_fake = disc(fake)
            loss_gen = -torch.nanmean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()

        scaler_gen.unscale_(opt_gen)
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 0.01)

        scaler_gen.step(opt_gen)
        scaler_gen.update()
        
        with torch.cuda.amp.autocast():
            alpha += cur_batch_size / (len(dataset) * config.PROGRESSIVE_EPOCHS[step]*0.5)
            alpha += config.SPECIAL_NUMBER
            alpha = min(alpha, 1)


        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE) * 0.5 + 0.5
                plot_to_tensorboard(
                    writer,
                    loss_disc.item(),
                    loss_gen.item(),
                    gp,
                    real.detach(),
                    fixed_fakes.detach(),
                    DfAg.DiffAugment(real),
                    DfAg.DiffAugment(fixed_fakes),
                    tensorboard_step,
                    now,
                    gen,
                    disc
                )
            tensorboard_step += 1

    if config.PROFILING:
        prof.stop()

    if config.SCHEDULER:
            scheduler_gen.step(loss_gen)
            scheduler_disc.step(loss_disc)

    
            
    print(f"Gradient Penalty: {gp.item()}\nAlpha: {alpha}")
    return tensorboard_step, alpha

def main():
    now = datetime.now()
    print(f"Versão do PyTorch: {torch.__version__}\nGPU utilizada: {torch.cuda.get_device_name(torch.cuda.current_device())}\nDataset: {config.DATASET}\nData-Horario: {now.strftime('%d/%m/%Y - %H:%M:%S')}")
    print(f"Profiling: {config.PROFILING}\n")
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
            opt_gen = optim.RMSprop(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, weight_decay=config.WEIGHT_DECAY, foreach=True)
            opt_disc = optim.RMSprop(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, weight_decay=config.WEIGHT_DECAY, foreach=True)
        case "ADAM":
            opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.0,0.99),weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.0,0.99),weight_decay=config.WEIGHT_DECAY)
        case "NADAM":
            opt_gen = optim.NAdam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.0,0.99), weight_decay=config.WEIGHT_DECAY)
            opt_disc = optim.NAdam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.0,0.99), weight_decay=config.WEIGHT_DECAY)
        case "ADAMAX":
            opt_gen = optim.Adamax(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.0,0.99), weight_decay=config.WEIGHT_DECAY, maximize=False)
            opt_disc = optim.Adamax(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.0,0.99), weight_decay=config.WEIGHT_DECAY, maximize=False)
        case "ADAMW":
            opt_gen = optim.AdamW(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.0,0.99), weight_decay=config.WEIGHT_DECAY, maximize=False)
            opt_disc = optim.AdamW(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.0,0.99), weight_decay=config.WEIGHT_DECAY, maximize=False)


    #Olhar mais a fundo
    schedulerGen = optim.lr_scheduler.ReduceLROnPlateau(
        opt_gen,
        patience=config.PATIENCE_DECAY,
        factor=0.4,
        min_lr=config.MIN_LEARNING_RATE, 
        verbose=True
        )
    schedulerDisc = optim.lr_scheduler.ReduceLROnPlateau(
        opt_disc,
        patience=config.PATIENCE_DECAY,
        factor=0.4,
        min_lr=config.MIN_LEARNING_RATE,
        verbose=True
        )

    scaler_disc = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(f"logs/LacadGan/{config.DATASET}/{now.strftime('%d-%m-%Y-%Hh%Mm%Ss')}")
    
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    cur_epoch = 0
    
    if config.LOAD_MODEL:
        epoch_s = [cur_epoch]
        step_s = [step]
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE_GENERATOR, epoch_s, step_s, config.DATASET
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, disc, opt_disc, config.LEARNING_RATE_DISCRIMINATOR, epoch_s, step_s, config.DATASET
        )
    
    gen.train()
    disc.train()
    
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4*2**step)
        img_size = 4*2**step
        print(f"Image size: {img_size}") 
        gen.set_alpha(alpha)
        gen.set_step(step)
        disc.set_alpha(alpha)
        disc.set_step(step)
        for epoch in range(cur_epoch,num_epochs):
            print(f"Epoch [{epoch}/{num_epochs}]")
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
        
            data_save_path = config.DATASET + "_"+ now.strftime("%d_%m_%Hh_%Mm")

            if config.GENERATE_IMAGES and (epoch%config.GENERATED_EPOCH_DISTANCE == 0) or epoch == (num_epochs-1) and config.GENERATE_IMAGES:
                img_generator = Thread(target=generate_examples, args=( gen, step, config.N_TO_GENERATE, (epoch-1), (4*2**step), data_save_path,), daemon=True)
                try:
                    img_generator.start()
                    img_generator.join()
                except Exception as err:
                    print(f"Erro: {err}")

            if config.SAVE_MODEL:
                gen_check = Thread(target=save_checkpoint, args=(gen, opt_gen, epoch, step, config.CHECKPOINT_GEN, data_save_path,), daemon=True)
                critic_check = Thread(target=save_checkpoint, args=(disc, opt_disc, epoch, step, config.CHECKPOINT_CRITIC, data_save_path,), daemon=True)
                try:
                    gen_check.start()
                    critic_check.start()
                    gen_check.join()
                    critic_check.join()
                except Exception as err:
                    print(f"Erro: {err}")
            torch.cuda.empty_cache()

        step += 1
        cur_epoch = 0


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision('high')
    
    with keepawake(keep_screen_awake=False):
        warnings.filterwarnings("ignore")
        path = str(pathlib.Path().resolve())
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', path, '--bind_all'])
        url = tb.launch()
        print(f"\n\nTensorboard rodando em {url}")
        main()
