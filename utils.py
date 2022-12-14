import torch
import random
import numpy as np
import os
import torchvision
import torch.nn as nn
import config
import pathlib
from torchvision.utils import save_image
from scipy.stats import truncnorm
import git

# Print losses occasionally and print to tensorboard
def plot_to_tensorboard(
    writer, loss_critic, loss_gen, real, fake, tensorboard_step
):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)

def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.reshape(gradient.shape[0], -1) # Troquei para reshape | era view
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def save_checkpoint(model, optimizer, epoch=0, step=0, filename="my_checkpoint.pth.tar", dataset="default"):
    caminho = str(pathlib.Path().resolve()) + "/imagens_geradas/" + dataset + "/" + filename
    print(f"=> Saving checkpoint in {filename}")
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, caminho)

def load_checkpoint(checkpoint_file, model, optimizer, lr, epoch, step, dataset="default"):

    caminho = str(pathlib.Path().resolve()) + "/imagens_geradas/" + dataset + "/" + checkpoint_file
    try:
        print(f"=> Loading checkpoint in {checkpoint_file}")
        checkpoint = torch.load(caminho, map_location="cuda")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch[0] = checkpoint["epoch"]
        step[0] = checkpoint["step"]

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    except Exception as exp:
        print("=> No checkpoint found")

def save_epoch_step(epoch=0,step=0,dataset="default",filename="/epoch_step.txt"): #Depreciada

    caminho = str(pathlib.Path().resolve()) + "/imagens_geradas/" + dataset + filename
    try:
        f = open(caminho,'w')
        f.write('{}\n'.format(epoch))
        f.write('{}'.format(step))
        f.close()
    except FileNotFoundError:
        print("=> Directory do not exist")

def load_epoch_step(dataset="default", filename="/epoch_step.txt"): #Depreciada
    caminho = str(pathlib.Path().resolve()) + "/imagens_geradas/" + dataset + filename
    try:
        f = open(caminho,'r')
        epoch = int(f.readline())
        step = int(f.readline())
        return epoch,step
    except FileNotFoundError:
        print("=> No epoch/step checkpoint found")
    except Exception as expt:
        print(expt)
    
    return 0,0

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_examples(gen, steps, truncation=0.7, n=100,epoch=0,size=0,name="default"):
    repo = git.Repo( str(pathlib.Path().resolve()) )
    print(repo.git.status())

    caminho = str(pathlib.Path().resolve()) + "/imagens_geradas/" + name
    if size < 10:
        parent_dir = caminho + "/size_0"+ str(size) +"/"
    else:
        parent_dir = caminho + "/size_"+ str(size) +"/"
        
    if os.path.isdir(parent_dir) == False:
        pathlib.Path(parent_dir).mkdir(parents=True, exist_ok=True)
        
    directory = "epoch_" + str(epoch + 1)
    path = os.path.join(parent_dir, directory)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(1, config.Z_DIM, 1, 1)), device=config.DEVICE, dtype=torch.float32)
            img = gen(noise, alpha, steps)
            save_image(img*0.5+0.5, f"{parent_dir}epoch_{epoch+1}/img_{i}.jpeg")

    gen.train()
