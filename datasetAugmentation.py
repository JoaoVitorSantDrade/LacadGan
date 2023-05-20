import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cuda
import torch.backends.cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import config
import os
import pathlib
import torch
from tqdm import tqdm
from datetime import datetime
import math

IMAGE = False
batchsize = 32
full_path = config.FOLDER_PATH + f"/Datasets/{config.DATASET}_aug/"

def get_loader(image_size):
    transform = transforms.Compose(
        [   
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
        ]
    )

    
    dataset = datasets.ImageFolder(root=f"Datasets/{config.DATASET}", transform=transform)   
    loader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return loader, dataset


transformation = torch.nn.Sequential(
    transforms.Normalize(
            [0.1 for _ in range(config.CHANNELS_IMG)], #antes era 0.5
            [0.9 for _ in range(config.CHANNELS_IMG)], #antes era 0.5
            ),
    transforms.RandomApply(
        (
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]),
        transforms.RandomVerticalFlip(),
        transforms.RandomErasing(scale=(0.01,0.04), ratio=(0.3,0.7)),
        ),
        p=0.5
    ),
    transforms.RandomApply(
        (
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]),
        transforms.RandomErasing(scale=(0.01,0.06), ratio=(0.3,0.9)),
        transforms.RandomHorizontalFlip(),
        ),
        p=0.5
    ),
    transforms.RandomAutocontrast(),
    transforms.RandomGrayscale(0.1),
    transforms.RandomInvert(0.1)
)

pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision('high')
    num_multiply = int(input("Number of times that you want to multiply your dataset: "))
    for step in range(int(math.log2(config.START_TRAIN_AT_IMG_SIZE/4)),config.SIMULATED_STEP):

        img_size = 4*2**step
        loop_master = tqdm(range(num_multiply), position=0, ncols=100,colour='blue', desc=f"{img_size}x{img_size}")
        for k, _ in enumerate(loop_master):
            pathlib.Path(full_path + f"/{img_size}x{img_size}/tensors_{k}").mkdir(parents=True, exist_ok=True)
            loader, _ = get_loader(img_size)
            loop = tqdm(loader, position=1, ncols=100, colour="Cyan", desc=f"Loop {k+1}", leave=False)
            for i, tensor in enumerate(loop):
                tensorGPU = tensor[0].to(config.DEVICE)
                tensorGPU = transformation(tensorGPU)
                for j in range(tensorGPU.shape[0]):
                    if IMAGE:
                        save_image(tensorGPU[j].clone(), f"Datasets/{config.DATASET}_aug/{img_size}x{img_size}/tensors_{k}/image{i + j + (i)*batchsize}.jpg")
                    else:
                        torch.save(tensorGPU[j].clone(), f"Datasets/{config.DATASET}_aug/{img_size}x{img_size}/tensors_{k}/tensor{i + j + (i)*batchsize}.pt")