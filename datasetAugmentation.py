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

batchsize = 16
full_path = str(pathlib.Path().resolve()) + f"/Datasets/{config.DATASET}-aug/"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((1536, 1536))
    ]
)

transformation = torch.nn.Sequential(
    transforms.RandomPosterize()
    #colocar as augmentations aqui
)

dataset = datasets.ImageFolder(root=f"Datasets/{config.DATASET}",transform=transform)

pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)


loader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

now = datetime.now()

loop = tqdm(loader, leave=True, smoothing=1)
for i, tensor in enumerate(loop):
    tensorGPU = tensor[0].to(config.DEVICE)
    tensorGPU = transformation(tensorGPU)
    for j, image in enumerate(tensorGPU):
        new_path = os.path.join(full_path,f"img{(i + j) + (i)*batchsize}-{now.strftime('%d-%m-%Y-%Hh%Mm%Ss')}.jpg")
        save_image(image,new_path)