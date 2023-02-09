from matplotlib.pyplot import sca
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import diffAugmentation as DfAg
from torch.utils.data import DataLoader
from threading import Thread

from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    load_epoch_step,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
    save_epoch_step,
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config


torch.backends.cudnn.benchmarks = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
            transforms.ColorJitter(brightness=0.1, hue=0.1),
        ]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=f"Datasets/{config.DATASET}", transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
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
    ):

    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        # Train Disc
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            
            if config.DIFF_AUGMENTATION:
                fake = DfAg.DiffAugment(fake)
                real = DfAg.DiffAugment(real)

            disc_real = disc(real, alpha, step)
            disc_fake = disc(fake.detach(), alpha, step)
            gp = gradient_penalty(disc, real, fake, alpha, step, device=config.DEVICE)
            loss_disc = (
                -(torch.mean(disc_real) - torch.mean(disc_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(disc_real ** 2))
            )

        opt_disc.zero_grad()
        scaler_disc.scale(loss_disc).backward()
        scaler_disc.step(opt_disc)
        scaler_disc.update()

        # Train Gen
        with torch.cuda.amp.autocast():
            gen_fake = disc(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        alpha += cur_batch_size / (len(dataset) * config.PROGRESSIVE_EPOCHS[step]*0.5)
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_disc.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1
    print(f"Loss Critic: {loss_disc}")
    return tensorboard_step, alpha

def main():
    print(f"Versão do PyTorch: {torch.__version__}\nGPU utilizada: {torch.cuda.get_device_name(torch.cuda.current_device())}\nDataset: {config.DATASET}")
    
    gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    disc = Discriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)

    #Initialize optmizer and scalers for FP16 Training
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.0,0.99),capturable=True)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.0,0.99),capturable=True)
    #opt_gen = optim.NAdam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.0,0.99))
    #opt_disc = optim.NAdam(disc.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(0.0,0.99))
    scaler_disc = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(f"logs/gan")

    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    cur_epoch = 0

    if config.LOAD_MODEL:
        #cur_epoch,step = load_epoch_step(dataset=config.DATASET) #Depreciada
        epoch_s = [cur_epoch]
        step_s = [step]
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE_GENERATOR, epoch_s, step_s, config.DATASET
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, disc, opt_disc, config.LEARNING_RATE_DISCRIMINATOR, epoch_s, step_s, config.DATASET
        )
    
    if not config.RESTART_LEARNING:
        cur_epoch = epoch_s[0]
        step = step_s[0]
        
    gen.train()
    disc.train()
    tensorboard_step = 0
    
    

    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4*2**step)
        img_size = 4*2**step
        print(f"Image size: {img_size}")
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
            )
            if config.GENERATE_IMAGES and (epoch%config.GENERATED_EPOCH_DISTANCE == 0) or epoch == (num_epochs-1):
                img_generator = Thread(target=generate_examples, args=( gen, step, config.N_TO_GENERATE, (epoch-1), (4*2**step), config.DATASET,), daemon=True)
                try:
                    img_generator.start()
                except Exception as err:
                    print(f"Erro: {err}")

            if config.SAVE_MODEL:
                #save_epoch_step(epoch=epoch,step=step,dataset=config.DATASET)                 #loss_disc loss_gen
                gen_check = Thread(target=save_checkpoint, args=(gen, opt_gen, epoch, step, config.CHECKPOINT_GEN, config.DATASET,), daemon=True)
                critic_check = Thread(target=save_checkpoint, args=(disc, opt_disc, epoch, step, config.CHECKPOINT_CRITIC, config.DATASET,), daemon=True)
                try:
                    gen_check.start()
                    critic_check.start()
                except Exception as err:
                    print(f"Erro: {err}")

        step += 1
        cur_epoch = 0
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()