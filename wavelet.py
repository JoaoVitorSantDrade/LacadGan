from pytorch_wavelets import DWTForward, DWTInverse

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

def upfirdn2d(input_tensor,filter,up_factor=1,down_factor=1,padding=(0,0,0,0)):
    r"""Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up_factor`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`filter`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down_factor`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        input_tensor:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        filter:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up_factor:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down_factor:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
    
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    image = F.upsample(input_tensor,mode="bilinear",scale_factor=up_factor)
    image = F.pad(image,pad=padding)
    image = F.conv2d(input=image,weight=filter)
    image = F.interpolate(image,mode="bilinear",scale_factor=down_factor)

    return image


class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer("ll", ll)
        self.register_buffer("lh", lh)
        self.register_buffer("hl", hl)
        self.register_buffer("hh", hh)

    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down_factor=2)
        lh = upfirdn2d(input, self.lh, down_factor=2)
        hl = upfirdn2d(input, self.hl, down_factor=2)
        hh = upfirdn2d(input, self.hh, down_factor=2)

        return torch.cat((ll, lh, hl, hh), 1)


class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer("ll", ll)
        self.register_buffer("lh", -lh)
        self.register_buffer("hl", -hl)
        self.register_buffer("hh", hh)

    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up_factor=2, padding=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up_factor=2, padding=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up_factor=2, padding=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up_factor=2, padding=(1, 0, 1, 0))

        return ll + lh + hl + hh
    
class WaveUpsampling(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.initial = nn.Sequential(
            InverseHaarTransform(in_channels),
            nn.Upsample(mode="bilinear",scale_factor=2),
            HaarTransform(in_channels)
        )

    def forward(self, x):
        out = self.initial(x)
        return out
    
class WaveDownsampling(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.initial = nn.Sequential(
            InverseHaarTransform(in_channels),
            F.interpolate(mode="bilinear",scale_factor=0.5),
            HaarTransform(in_channels)
        )

    def forward(self, x):
        out = self.initial(x)
        return out

