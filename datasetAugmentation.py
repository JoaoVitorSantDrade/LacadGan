import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import config
import os
import pathlib
import torch
from tqdm import tqdm
from datetime import datetime

torch.backends.cudnn.benchmarks = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_float32_matmul_precision('medium')

batchsize = 128
full_path = str(pathlib.Path().resolve()) + f"/Datasets/{config.DATASET}_aug/"

def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)], #antes era 0.5
                [0.5 for _ in range(config.CHANNELS_IMG)], #antes era 0.5
            ),
        ]
    )

    
    dataset = datasets.ImageFolder(root=f"Datasets/{config.DATASET}", transform=transform)   
    loader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    return loader, dataset

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)], #antes era 0.5
                [0.5 for _ in range(config.CHANNELS_IMG)], #antes era 0.5
            ),
    ]
)

transformation = torch.nn.Sequential(
    #transforms.RandomPosterize()
    #colocar as augmentations aqui
)

dataset = datasets.ImageFolder(root=f"Datasets/{config.DATASET}",transform=transform)

pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)


loader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=16
    )

if __name__ == "__main__":
    for step in range(config.SIMULATED_STEP):
        img_size = 4*2**step
        pathlib.Path(full_path + f"/{img_size}x{img_size}/tensors").mkdir(parents=True, exist_ok=True)
        loader, dataset = get_loader(img_size)
        loop = tqdm(loader, leave=False, smoothing=1, miniters=1, unit_scale=True)
        #total_steps = len(loop)*factors[step]
        print(f"\nData augmentation: {img_size}x{img_size}")
        for i, tensor in enumerate(loop):
            tensorGPU = tensor[0].to(config.DEVICE)
            tensorGPU = transformation(tensorGPU)
            for j in range(tensorGPU.shape[0]):
                torch.save(tensorGPU[j].clone(), f"Datasets/{config.DATASET}_aug/{img_size}x{img_size}/tensors/tensor{i + j + (i)*batchsize}.pt")
            #for j, image in enumerate(tensorGPU):
                #new_path = os.path.join(full_path,f"img{(i + j) + (i)*batchsize}-{now.strftime('%d-%m-%Y-%Hh%Mm%Ss')}.pt")
                #torch.save(image.clone(),f"Datasets/{config.DATASET}_aug/{img_size}x{img_size}/tensors/tensor{(i + j) + (i)*batchsize}-{now.strftime('%d-%m-%Y-%Hh%Mm%Ss')}.pt")