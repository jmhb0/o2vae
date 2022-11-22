import torch.nn as nn 
import torchvision
import numpy as np

class CnnDecoder(nn.Module):
    def __init__(self, zdim, cnn_dims = [32,64,128,256], out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1,
            last_kernel_size=None,):
        """
        Args:
            last_kernel_size (int): allows to control the very last conv layer before output 
        """
        super().__init__()
        activation = nn.ELU()
        self.fc = nn.Sequential(
            nn.Linear(zdim, cnn_dims[-1]),
            nn.ELU(),
        )

        layers=[]
        cnn_dims = np.flip(np.array(cnn_dims))
        in_dim = cnn_dims[0]
        
        for i, dim in enumerate(cnn_dims[1:]):
            layers.append(nn.ConvTranspose2d(in_dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))
            layers.append(nn.BatchNorm2d(dim))
            layers.append(activation)
            in_dim=dim

        if last_kernel_size is None: last_kernel_size=kernel_size
        layers.append(nn.ConvTranspose2d(in_dim, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                         padding=padding, output_padding=output_padding))

        self.dec_conv = nn.Sequential(*layers)

    def forward(self, x):
        bs=x.size(0)
        x = self.fc(x)
        dim = x.size(1)
        x = x.view(bs, dim, 1, 1)
        x = self.dec_conv(x)

        return  x