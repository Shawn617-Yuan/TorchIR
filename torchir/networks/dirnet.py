from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn.modules.utils import _ntuple


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        downsample=(False,),
        # Activation function
        af=nn.ELU,
        ndim=2,
    ):
        # If ndim is 2, nn.Conv2d is chosen. If ndim is 3, nn.Conv3d is chosen
        Conv = (nn.Conv2d, nn.Conv3d)[ndim - 2]
        AvgPool = (nn.AvgPool2d, nn.AvgPool3d)[ndim - 2]

        # To ensure that the output of the convolutional layer has the same spatial dimensions as its input, 
        # the padding is set to half of the kernel size. This assumes a stride of 1.
        padding = kernel_size // 2

        # Keep track of the layers in the order they are added.
        layers = OrderedDict()
        layers["conv"] = Conv(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        if af:
            layers["af"] = af()
        if any(downsample):
            if any(np.asarray(downsample) > 1):  # TODO: improve checking
                layers["downsample"] = AvgPool(tuple(downsample))
        
        # The parent class "nn.Sequential" class is a container in PyTorch that can hold multiple modules 
        # (like convolutional layers, activation functions, etc.) and process them in a sequential manner
        super().__init__(layers)


class DIRNet(nn.Module):
    def __init__(
        self,
        grid_spacing,
        kernel_size=3,
        kernels=32,
        num_conv_layers=5,
        num_dense_layers=2,
        ndim=2,
    ):
        # Provides essential methods and attributes for custom neural network modules in PyTorch.
        super().__init__()
        
        # "_ntuple" is used to ensure that the grid_spacing is a tuple with the length ndim
        grid_spacing = _ntuple(ndim)(grid_spacing)
        self.grid_spacing = grid_spacing

        AF = torch.nn.ELU

        num_downsamplings = np.log2(grid_spacing)
        
        # Ensure that the grid_spacing values are powers of 2 by checking if the logarithm values are whole numbers.
        assert all(
            num_downsamplings == num_downsamplings.round()
        ), "Grid spacing should be factors of 2."
        num_downsamplings = num_downsamplings.astype(int)

        # 2 stand for fixed and moving images
        in_channels = 2
        downsample = (0 < num_downsamplings).astype(np.int) + 1  # downsample kernel
        conv_layers = [
            ConvBlock(
                in_channels,
                kernels,
                kernel_size=kernel_size,
                downsample=downsample,
                af=AF,
                ndim=ndim,
            )
        ]
        for i in range(1, num_conv_layers):
            downsample = (i < num_downsamplings).astype(np.int) + 1  # downsample kernel
            conv_layers.append(
                ConvBlock(
                    kernels,
                    kernels,
                    kernel_size=kernel_size,
                    downsample=downsample,
                    af=AF,
                    ndim=ndim,
                )
            )

        # Convolutional layers with a kernel size of 1 (dense layers)
        dense_layers = [
            ConvBlock(kernels, kernels, kernel_size=1, af=AF, ndim=ndim)
            for _ in range(num_dense_layers)
        ]
        dense_layers.append(
            ConvBlock(kernels, ndim, kernel_size=1, af=None, ndim=ndim)
        )  # linear activation for b-spline coeffs.
        
        # The last dense layer outputs ndim channels with initialized weights and biases set to zero, 
        # probably to predict coefficients for b-splines.
        dense_layers[-1].conv.weight.data.fill_(0.0)
        dense_layers[-1].conv.bias.data.fill_(0.0)
        
        # The convolutional and dense layers are then packed into nn.Sequential containers for convenient forward propagation.
        self.conv_layers = nn.Sequential(*conv_layers)
        self.dense_layers = nn.Sequential(*dense_layers)
        self.ndim = ndim

    def forward(self, fixed, moving=None):
        # concatenated with the fixed image along the channel dimension.
        x = fixed
        if moving is not None:
            x = torch.cat((fixed, moving), 1)

        x = self.conv_layers(x)
        x = self.dense_layers(x)

        return x
