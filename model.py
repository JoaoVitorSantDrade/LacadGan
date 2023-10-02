import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2, sqrt
import config
import pytorch_wavelets as wave
import numpy as np
import matplotlib.pyplot as plt
from millify import millify


factors = [1, 1, 1, 1, 1, 1, 1, 1, 1] #factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

#https://blog.paperspace.com/implementation-stylegan2-from-scratch/

def get_w(batch_size, mapping_network):

    z = torch.randn(batch_size, config.W_DIM).to(config.DEVICE)
    w = mapping_network(z)
    w = w[None, :, :].expand(config.SIMULATED_STEP, -1, -1)
    return w

def get_noise(batch_size):
    
        noise = []
        resolution = 4

        for i in range(config.SIMULATED_STEP):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=config.DEVICE)
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=config.DEVICE)

            noise.append((n1, n2))

            resolution *= 2

        return noise

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim)
        )

    def forward(self, x):
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)  # for PixelNorm 
        return self.mapping(x)

class Generator(nn.Module):

    def __init__(self, log_resolution, W_DIM, n_features = 32, max_features = 256):

        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(W_DIM, features[0], features[0])
        self.to_rgb = ToRGB(W_DIM, features[0])

        blocks = [GeneratorBlock(W_DIM, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, w, input_noise):

        batch_size = w.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])

        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new

        return torch.tanh(rgb)

class RGBtoWave(nn.Module):

    def __init__(self):

        super().__init__()
        self.DWT = wave.DWTForward(J=1,wave="haar")

        self.conv2D = nn.Conv2d(3,3,kernel_size=1)
        self.conv3D = nn.Conv3d(3,3,kernel_size=1) # Não alterar o kernel size
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = x.to(torch.float)
        yl, yh = self.DWT(x)
        yl = self.act(self.conv2D(yl))
        yh_new = self.act(self.conv3D(yh[0]))
        return (yl,[yh_new])

class WaveGenerator(nn.Module):

    def __init__(self, log_resolution, W_DIM, n_features = 32, max_features = 256):

        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(W_DIM, features[0], features[0])
        self.to_rgb = ToRGB(W_DIM, features[0])

        blocks = [WaveGeneratorBlock(W_DIM, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

        self.composition_wave = wave.DWTInverse(wave="haar")

        self.final_activation = torch.nn.LeakyReLU(0.1)
    def forward(self, w, input_noise):

        batch_size = w.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])
        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new, wavelets_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + self.composition_wave(wavelets_new) + rgb_new
        return self.final_activation(rgb)

class WaveGeneratorBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):

        super().__init__()

        self.style_block1 = StyleBlock(W_DIM, in_features, out_features)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features)

        self.to_rgb = ToRGB(W_DIM, out_features)
        self.to_wave = RGBtoWave()

    def forward(self, x, w, noise):

        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)
        wavelets = self.to_wave(rgb)

        return x, rgb, wavelets

class GeneratorBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):

        super().__init__()

        self.style_block1 = StyleBlock(W_DIM, in_features, out_features)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features)

        self.to_rgb = ToRGB(W_DIM, out_features)
        self.to_wave = RGBtoWave()

    def forward(self, x, w, noise):

        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb

class StyleBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):

        super().__init__()

        self.to_style = EqualizedLinear(W_DIM, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):

        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])

class ToRGB(nn.Module):

    def __init__(self, W_DIM, features):

        super().__init__()
        self.to_style = EqualizedLinear(W_DIM, features, bias=1.0)

        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):

        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])

class FromRGB(nn.Module):   
    def __init__(self, n_features):
        super().__init__()
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1), 
            nn.LeakyReLU(0.2, True),)
        
    def forward(self, x):
        x = self.from_rgb(x)
        return x

class Conv2dWeightModulate(nn.Module):

    def __init__(self, in_features, out_features, kernel_size,
                 demodulate = True, eps = 1e-8):

        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x, s):

        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)

class WaveDiscriminator(nn.Module):

    def __init__(self, log_resolution, n_features = 64, max_features = 256):

        super().__init__()

        self.dwt = wave.DWTForward(J=1, wave="haar", mode="zero")

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]

        self.from_rgb = FromRGB(n_features)

        n_blocks = len(features) - 1
        blocks = [WaveDiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_features = features[-1] + 1
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

        self.to_wave = RGBtoWave()
        self.composition_image = wave.DWTInverse(wave="haar")

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x):

        wavelet = self.to_wave(x)
        wave_img = self.composition_image(wavelet) #Tenho que criar um bloco só pra wavelet
        x = self.from_rgb(x + wave_img)
        
        x = self.blocks(x)

        x = self.minibatch_std(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)

class Discriminator(nn.Module):

    def __init__(self, log_resolution, n_features = 64, max_features = 256):

        super().__init__()

        self.dwt = wave.DWTForward(J=1, wave="haar", mode="zero")

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]

        self.from_rgb = FromRGB(n_features)

        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_features = features[-1] + 1
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x):

        x = self.from_rgb(x)
        x = self.blocks(x)

        x = self.minibatch_std(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)

class WaveDiscriminatorBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), # down sampling using avg pool
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        #Em implementação
        self.to_Wave = RGBtoWave()
        self.composition_wave = wave.DWTInverse(wave="haar")

        self.scale = 1 / sqrt(2) # 0.707

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)
                
        x = self.down_sample(x)

        return (x + residual) * self.scale

class DiscriminatorBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), # down sampling using avg pool
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        self.scale = 1 / sqrt(2) # 0.707

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale

class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias = 0.1):

        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)

class EqualizedConv2d(nn.Module):

    def __init__(self, in_features, out_features,
                 kernel_size, padding = 0):

        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)

class EqualizedWeight(nn.Module):

    def __init__(self, shape):

        super().__init__()

        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c

class PathLengthPenalty(nn.Module):

    def __init__(self, beta):

        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):

        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / sqrt(image_size)
        sqrt(image_size)

        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:

            a = self.exp_sum_a / (1 - self.beta ** self.steps)

            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss


#https://colab.research.google.com/drive/1dAF1_oAQ1c8OMLqlYA5V878pmpcnQ6_9?usp=sharing#scrollTo=1q1iycR5h9jh
"""
##### Copyright 2021 Mahmoud Afifi.

 If you find this code useful, please cite our paper:

 Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN: 
 Controlling Colors of GAN-Generated and Real Images via Color Histograms." 
 In CVPR, 2021.

 @inproceedings{afifi2021histogan,
  title={Histo{GAN}: Controlling Colors of {GAN}-Generated and Real Images via 
  Color Histograms},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={CVPR},
  year={2021}
}
####
"""
EPS = 1e-6
class RGBuvHistBlock(nn.Module):
  def __init__(self, h=64, insz=150, resizing='interpolation',
               method='inverse-quadratic', sigma=0.02, intensity_scale=True,
               hist_boundary=None, green_only=False, device='cuda'):
    """ Computes the RGB-uv histogram feature of a given image.
    Args:
      h: histogram dimension size (scalar). The default value is 64.
      insz: maximum size of the input image; if it is larger than this size, the 
        image will be resized (scalar). Default value is 150 (i.e., 150 x 150 
        pixels).
      resizing: resizing method if applicable. Options are: 'interpolation' or 
        'sampling'. Default is 'interpolation'.
      method: the method used to count the number of pixels for each bin in the 
        histogram feature. Options are: 'thresholding', 'RBF' (radial basis 
        function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
      sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is 
        the sigma parameter of the kernel function. The default value is 0.02.
      intensity_scale: boolean variable to use the intensity scale (I_y in 
        Equation 2). Default value is True.
      hist_boundary: a list of histogram boundary values. Default is [-3, 3].
      green_only: boolean variable to use only the log(g/r), log(g/b) channels. 
        Default is False.

    Methods:
      forward: accepts input image and returns its histogram feature. Note that 
        unless the method is 'thresholding', this is a differentiable function 
        and can be easily integrated with the loss function. As mentioned in the
         paper, the 'inverse-quadratic' was found more stable than 'RBF' in our 
         training.
    """
    super(RGBuvHistBlock, self).__init__()
    self.h = h
    self.insz = insz
    self.device = device
    self.resizing = resizing
    self.method = method
    self.intensity_scale = intensity_scale
    self.green_only = green_only
    if hist_boundary is None:
      hist_boundary = [-3, 3]
    hist_boundary.sort()
    self.hist_boundary = hist_boundary
    if self.method == 'thresholding':
      self.eps = (abs(hist_boundary[0]) + abs(hist_boundary [1])) / h
    else:
      self.sigma = sigma
    

  def forward(self, x):
    x = torch.clamp(x, 0, 1)
    if x.shape[2] > self.insz or x.shape[3] > self.insz:
      if self.resizing == 'interpolation':
        x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                  mode='bilinear', align_corners=False)
      elif self.resizing == 'sampling':
        inds_1 = torch.LongTensor(
          np.linspace(0, x.shape[2], self.h, endpoint=False)).to(
          device=self.device)
        inds_2 = torch.LongTensor(
          np.linspace(0, x.shape[3], self.h, endpoint=False)).to(
          device=self.device)
        x_sampled = x.index_select(2, inds_1)
        x_sampled = x_sampled.index_select(3, inds_2)
      else:
        raise Exception(
          f'Wrong resizing method. It should be: interpolation or sampling. '
          f'But the given value is {self.resizing}.')
    else:
      x_sampled = x

    L = x_sampled.shape[0]  # size of mini-batch
    if x_sampled.shape[1] > 3:
      x_sampled = x_sampled[:, :3, :, :]
    X = torch.unbind(x_sampled, dim=0)
    hists = torch.zeros((x_sampled.shape[0], 1 + int(not self.green_only) * 2, 
                         self.h, self.h)).to(device=self.device)
    for l in range(L):
      I = torch.t(torch.reshape(X[l], (3, -1)))
      II = torch.pow(I, 2)
      if self.intensity_scale:
        Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS), 
                             dim=1)
      else:
        Iy = 1
      if not self.green_only:
        Iu0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 1] + 
                                                                   EPS), dim=1)
        Iv0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 2] + 
                                                                   EPS), dim=1)
        diff_u0 = abs(
          Iu0 - torch.unsqueeze(torch.tensor(np.linspace(
              self.hist_boundary[0], self.hist_boundary[1], num=self.h)), 
              dim=0).to(self.device))
        diff_v0 = abs(
          Iv0 - torch.unsqueeze(torch.tensor(np.linspace(
              self.hist_boundary[0], self.hist_boundary[1], num=self.h)), 
              dim=0).to(self.device))
        if self.method == 'thresholding':
          diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
          diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
        elif self.method == 'RBF':
          diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_u0 = torch.exp(-diff_u0)  # Radial basis function
          diff_v0 = torch.exp(-diff_v0)
        elif self.method == 'inverse-quadratic':
          diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
          diff_v0 = 1 / (1 + diff_v0)
        else:
          raise Exception(
            f'Wrong kernel method. It should be either thresholding, RBF,' 
            f' inverse-quadratic. But the given value is {self.method}.')
        diff_u0 = diff_u0.type(torch.float32)
        diff_v0 = diff_v0.type(torch.float32)
        a = torch.t(Iy * diff_u0)
        hists[l, 0, :, :] = torch.mm(a, diff_v0)

      Iu1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 0] + EPS),
                            dim=1)
      Iv1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 2] + EPS),
                            dim=1)
      diff_u1 = abs(
        Iu1 - torch.unsqueeze(torch.tensor(np.linspace(
            self.hist_boundary[0], self.hist_boundary[1], num=self.h)), 
            dim=0).to(self.device))
      diff_v1 = abs(
        Iv1 - torch.unsqueeze(torch.tensor(np.linspace(
            self.hist_boundary[0], self.hist_boundary[1], num=self.h)), 
            dim=0).to(self.device))

      if self.method == 'thresholding':
        diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
        diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
      elif self.method == 'RBF':
        diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u1 = torch.exp(-diff_u1)  # Gaussian
        diff_v1 = torch.exp(-diff_v1)
      elif self.method == 'inverse-quadratic':
        diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
        diff_v1 = 1 / (1 + diff_v1)

      diff_u1 = diff_u1.type(torch.float32)
      diff_v1 = diff_v1.type(torch.float32)
      a = torch.t(Iy * diff_u1)
      if not self.green_only:
        hists[l, 1, :, :] = torch.mm(a, diff_v1)
      else:
        hists[l, 0, :, :] = torch.mm(a, diff_v1)

      if not self.green_only:
        Iu2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 0] + 
                                                                   EPS), dim=1)
        Iv2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 1] + 
                                                                   EPS), dim=1)
        diff_u2 = abs(
          Iu2 - torch.unsqueeze(torch.tensor(np.linspace(
              self.hist_boundary[0], self.hist_boundary[1], num=self.h)), 
              dim=0).to(self.device))
        diff_v2 = abs(
          Iv2 - torch.unsqueeze(torch.tensor(np.linspace(
              self.hist_boundary[0], self.hist_boundary[1], num=self.h)), 
              dim=0).to(self.device))
        if self.method == 'thresholding':
          diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
          diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
        elif self.method == 'RBF':
          diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_u2 = torch.exp(-diff_u2)  # Gaussian
          diff_v2 = torch.exp(-diff_v2)
        elif self.method == 'inverse-quadratic':
          diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                              2) / self.sigma ** 2
          diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
          diff_v2 = 1 / (1 + diff_v2)
        diff_u2 = diff_u2.type(torch.float32)
        diff_v2 = diff_v2.type(torch.float32)
        a = torch.t(Iy * diff_u2)
        hists[l, 2, :, :] = torch.mm(a, diff_v2)

    # normalization
    hists_normalized = hists / (
        ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

    return hists_normalized

if __name__ == "__main__":

    gen = WaveGenerator(config.SIMULATED_STEP,config.W_DIM).to(config.DEVICE)
    critic = WaveDiscriminator(config.SIMULATED_STEP, n_features=128).to(config.DEVICE)
    map = MappingNetwork(config.Z_DIM,config.W_DIM).to(config.DEVICE)
    
    w = get_w(1, map)
    noise = get_noise(1)

    
    with torch.cuda.amp.autocast():
        fake = gen(w, noise)
        critic_fake = critic(fake.detach())
        
    assert fake.shape == (1, 3, (2**config.SIMULATED_STEP), (2**config.SIMULATED_STEP))
    assert critic_fake.shape == (1,1)

    print(f"Success! At img size: {fake.shape[2]} - step: {config.SIMULATED_STEP}")

    gen_total_params = sum(p.numel() for p in gen.parameters() if p.requires_grad) 
    critic_total_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    map_total_params = sum(p.numel() for p in map.parameters() if p.requires_grad)

    print(f"\nGen: {millify(gen_total_params)}\nCritic: {millify(critic_total_params)}\nMap: {millify(map_total_params)}\n")