import torch
import random
import numpy as np
import os
import torchvision
import config
import pathlib
from torchvision.utils import save_image
from scipy.stats import truncnorm
from torch.utils.tensorboard import SummaryWriter
from model import get_w, get_noise
import shutil
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure,FrechetInceptionDistance
import asyncio
from math import log2
import torchviz as tvz
from torchview import draw_graph

mem_format = torch.channels_last if config.CHANNEL_LAST == True else torch.contiguous_format

if config.MS_SSIM:
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(config.DEVICE,non_blocking=True,memory_format = mem_format)

if config.SSIM:
    ssim = StructuralSimilarityIndexMeasure().to(config.DEVICE,non_blocking=True,memory_format = mem_format)

if config.PSNR:
    psnr = PeakSignalNoiseRatio().to(config.DEVICE,non_blocking=True,memory_format = mem_format)

if config.FID:
    fid = FrechetInceptionDistance().to(device=config.DEVICE, non_blocking=True,memory_format = mem_format)

def write_to_save_path(filename,item,now, unit = ""):
    try:
        with open(f'saves/{config.DATASET + "_" + now.strftime("%d-%m-%Y-%Hh%Mm%Ss") + "_" + config.OPTMIZER}/{filename}.txt','a') as f:
            f.write(f"{filename}: {item} {unit}\n")
    except:
        pathlib.Path(config.FOLDER_PATH + "/saves/" + config.DATASET + "_"+ now.strftime("%d-%m-%Y-%Hh%Mm%Ss")  + "_" + config.OPTMIZER + "/").mkdir(parents=True, exist_ok=True)
        with open(f'saves/{config.DATASET + "_"+ now.strftime("%d-%m-%Y-%Hh%Mm%Ss") + "_" + config.OPTMIZER}/{filename}.txt','a') as f:
            f.write(f"{filename}: {item} {unit}\n")

def calculate_mifid(true_images,false_images,now):
    #https://arxiv.org/pdf/2106.03062.pdf
    pass

def calculate_psnr(true_images,false_images,now):
    #Quanto maior, melhor
    grey_scale = torchvision.transforms.Grayscale()
    psnr_score = psnr(grey_scale(true_images),grey_scale(false_images))
    write_to_save_path("psnr_score",(psnr_score.detach()),now,unit="DBs")
    return 1 / torch.log10(psnr_score)

def calculate_ssim(true_images,false_images,now):
    grey_scale = torchvision.transforms.Grayscale()
    #https://arxiv.org/pdf/2006.13846.pdf
    #https://repositorio.ufes.br/handle/10/9659

    ssim_score = ssim(grey_scale(false_images), grey_scale(true_images))
    write_to_save_path("ssim_score",ssim_score.detach(),now)
    return log2(2 - ssim_score)

def calculate_ms_ssim(true_images,false_images,now):
    #https://torchmetrics.readthedocs.io/en/stable/image/multi_scale_structural_similarity.html
    
    grey_scale = torchvision.transforms.Grayscale()
    ms_ssim_score = ms_ssim(grey_scale(false_images), grey_scale(true_images))
    write_to_save_path("ms_ssim_score",ms_ssim_score.detach(),now)
    return log2(2 - ms_ssim_score) 

def calculate_fid(true_images,false_images, now):
    true_images = true_images.to(torch.uint8)
    false_images = false_images.to(torch.uint8)

    fid.update(true_images,True)
    fid.update(false_images,False)
    fid_score = fid.compute()
    write_to_save_path("fid_score",fid_score.detach(),now)
    return fid_score

# Print losses occasionally and print to tensorboard
def plot_to_tensorboard(
    writer:SummaryWriter, loss_critic, loss_gen, real, fake, tensorboard_step, now, gen, critic, gp
):
    # precision = true_p/(true_p+false_p+config.SPECIAL_NUMBER)
    # recall = true_p/(true_p+false_n+config.SPECIAL_NUMBER)
    # writer.add_scalars("precision_recall", {"precision":precision, "recall":recall}, global_step=tensorboard_step)
    writer.add_scalar("loss/critic", loss_critic,global_step=tensorboard_step,new_style=True)
    writer.add_scalar("loss/gen", loss_gen,global_step=tensorboard_step,new_style=True)
    writer.add_scalar("loss/distance", abs(loss_gen) + abs(loss_critic),global_step=tensorboard_step,new_style=True)

    writer.add_scalar("GradientPenalty/", gp, global_step=tensorboard_step, new_style=True)
    # for name, param in gen.named_parameters():
    #     if param.requires_grad and "weight" in name:
    #         writer.add_histogram(f"generator/{name}",param.data,global_step=tensorboard_step)
    # for name, param in critic.named_parameters():
    #     if param.requires_grad and "weight" in name:
    #         writer.add_histogram(f"critic/{name}",param.data,global_step=tensorboard_step)

    with torch.no_grad():
        img_grid_real = torchvision.utils.make_grid(real[:8])
        img_grid_fake = torchvision.utils.make_grid(fake[:8])
        if config.LOAD_MODEL:
            aux = f"{config.WHERE_LOAD}"
            aux = aux.replace(f"{config.DATASET}_","")
            writer.add_image(f'Real/{config.DATASET}-{aux}',img_grid_real, global_step=tensorboard_step)
            writer.add_image(f'Fake/{config.DATASET}-{aux}',img_grid_fake, global_step=tensorboard_step)
        else:
            writer.add_image(f'Real/{config.DATASET}-{now.strftime("%d-%m-%Y-%Hh%Mm%Ss")}',img_grid_real, global_step=tensorboard_step)
            writer.add_image(f'Fake/{config.DATASET}-{now.strftime("%d-%m-%Y-%Hh%Mm%Ss")}',img_grid_fake, global_step=tensorboard_step)
        if config.VIDEO:
            save_image(fake,pathlib.Path(config.FOLDER_PATH + f"/images/img_{tensorboard_step}.jpeg"))
    return

def remove_graphs():
    path = pathlib.Path(str(pathlib.Path().resolve()) + f"/logs/LacadGan")
    for files in os.listdir(path):
        if os.path.isfile(os.path.join(path, files)):
            if "gen" or "critic" in files:
                file_path = pathlib.Path.joinpath(path,files)
                shutil.rmtree(file_path)

def plot_cnns_tensorboard(gen, disc, noise, weight=None):

    for step in range(config.SIMULATED_STEP):
        writerGen = SummaryWriter(f"logs/LacadGan/{2**(step+1) * 4}x{2**(step+1) * 4}_gen")
        writerCritc = SummaryWriter(f"logs/LacadGan/{2**(step+1) * 4}x{2**(step+1) * 4}_critic")
        gen.eval()
        disc.eval()
        with torch.no_grad():
            writerGen.add_graph(gen,weight, noise,verbose = False, use_strict_trace=False)
            writerCritc.add_graph(disc,gen(weight, noise),verbose = False, use_strict_trace=False)
        writerGen.close()
        writerCritc.close()
        gen.train()
        disc.train()
        
def gradient_penalty(critic, real, fake):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(config.DEVICE)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    match config.MODEL:
        case "Style":
            mixed_scores = critic(interpolated_images)
        case "Pro":
            mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores,device=config.DEVICE),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.reshape((BATCH_SIZE, -1)) # Troquei para reshape | era view
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def save_checkpoint(model, optimizer,scheduler=None, epoch=0, step=0, filename="my_checkpoint.pth.tar", dataset="default"):
    caminho = config.FOLDER_PATH + "/saves/" + dataset + "/" + filename
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint.update({"scheduler":scheduler})
    try:
        torch.save(checkpoint, caminho)
    except:
        pathlib.Path(config.FOLDER_PATH + "/saves/" + dataset + "/").mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, caminho)
    return 

def load_checkpoint(checkpoint_file, model, optimizer=None, epoch=0, step=0, scheduler=None, dataset="default", inference=False):

    caminho = config.FOLDER_PATH + "/saves/" + dataset + "/" + checkpoint_file
    try:
        print(f"=> Loading checkpoint in {checkpoint_file}")
        checkpoint = torch.load(caminho, map_location="cuda")
        model.load_state_dict(checkpoint["state_dict"])
        if not inference:
            optimizer.load_state_dict(checkpoint["optimizer"])
            epoch[0] = checkpoint["epoch"]
            step[0] = checkpoint["step"]
            if config.SCHEDULER:
                scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"=> Success loading {checkpoint_file}")
    except Exception as exp:
        print(f"=> No checkpoint found in {checkpoint_file} - {exp}")
   

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_examples(gen, mapping_network, steps, n=1000,epoch=0,size=0,name="default", critic = None):
    truncation = 0.7
    caminho = config.FOLDER_PATH + "/saves/" + name
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
            match config.MODEL:
                case "Pro":
                    gen.set_alpha(alpha)
                    gen.set_step(steps)

            if config.VIDEO:
                noise = config.FIXED_NOISE[0]
                img = gen(noise)

            match config.MODEL:
                case "Style":
                    w     = get_w(1, mapping_network)
                    noise = get_noise(1)
                    img = gen(w, noise)
                case "Pro":
                    noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(1, config.Z_DIM, 1, 1)), device=config.DEVICE, dtype=torch.float32)
                    noise = noise.to(memory_format=torch.channels_last)
                    img = gen(noise)

            save_image(img*0.5+0.5, f"{parent_dir}epoch_{epoch+1}/img_{i}.jpeg")
    return

def draw_model_graph(name,map,gen,critic=None):
        w     = get_w(1, map)
        noise = get_noise(1)
        img = gen(w, noise)

        parent_dir = config.FOLDER_PATH + "/saves/" + name
        if os.path.isdir(parent_dir) == False:
            pathlib.Path(parent_dir).mkdir(parents=True, exist_ok=True)

        torchview = draw_graph(
            model=gen,
            input_data=(w,noise),
            directory=parent_dir,
            roll=True,
            expand_nested=True,
            )
        torchview.visual_graph.render(f"Gen",format="png",directory=parent_dir)

        if critic:
            torchview = draw_graph(
            model=critic,
            input_data=img,
            directory=parent_dir,
            roll=True,
            expand_nested=True,
            )
            torchview.visual_graph.render(f"Critic",format="png",directory=parent_dir)
        
        
def generate_graph(model, *model_input, path_to_save, name=None):
    dots = tvz.make_dot(model(model_input), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    dots.render(f"{name}_Graph", format="png",directory=path_to_save)

def show_loaded_model():
    model = torch.load(str(pathlib.Path().resolve()) + "/saves/" + config.DATASET +"/"+ config.CHECKPOINT_CRITIC)
    print(model["state_dict"])

if __name__ == "__main__":
    plot_cnns_tensorboard()