import torch
from torchvision.utils import make_grid
from models import align_reconstructions
import torchvision.transforms as T
import torchvision.transforms.functional as T_f
import numpy as np

def grid_from_2cols(x1, x2, nrow=10, ncol=8, 
        grid_kwargs={'padding':2, 'pad_value':0.5, 'align_corners':False}):
    """ 
    Given 2 tensors of images x1, x2, put them in a single imagegrid (by 
    calling torchvision.utils.make_grid) where x1[i] is next to x2[i]. 
    It's useful for plotting an original image and its reconstruction. 
    """
    n = nrow*(ncol//2)
    x1, x2 = x1[:n], x2[:n]
    # add gray bars to separate the columns 
    gray_bar = 0.2*torch.ones((*x1.shape[:3], 2))
    x1 = torch.cat((gray_bar, x1), dim=3)
    x2 = torch.cat((x2, gray_bar), dim=3)

    # merge the images and create a grid
    assert x1.shape==x2.shape
    x = torch.zeros((len(x1)*2, *x1.shape[1:]), dtype=x1.dtype)
    x[0::2] = x1
    x[1::2] = x2
    grid = make_grid(x, ncol, **grid_kwargs)
    grid = torch.permute(grid, (1,2,0))
    return grid

def grid_from_3cols(x1, x2, x3, nrow=10, ncol=6):
    """
    Same as `grid_from_2cols` but for 3 columns.
    """
    n = nrow*(ncol//3)
    x1, x2, x3 = x1[:n], x2[:n], x3[:n]

    assert x1.shape==x2.shape
    x = torch.zeros((len(x1)*3, *x1.shape[1:]), dtype=x1.dtype)
    x[0::3] = x1
    x[1::3] = x2
    x[2::3] = x3
    grid = make_grid(x, ncol)
    grid = torch.permute(grid, (1,2,0))
    return

def reconstruction_grid(model, x, align=True, nrow=12, ncol=8, device='cuda'):
    """
    Run on cpu, and non-efficient loops, because we assume it's  not run too often.
    Given a trained model and batch of data with shape (N,C,Y,X), return a tensor
    that can be plotted as an image using matplotlib imshow().
    grid that can 
    where one column is
    """
    assert len(x)<=256
    model.eval().to(device)
    with torch.no_grad():
        y=model.reconstruct(x.to(device)).cpu()

    # rotate and flip y to align with x
    if align: 
        # array `align_transforms` is the optimal rotation and flip
        _, align_transforms = align_reconstructions.loss_reconstruction_fourier_batch(
            x,y, recon_loss_type=model.loss_kwargs['recon_type']
        )
        # flip and rotate
        idxs_flip = np.where(align_transforms[:,1])[0]
        y[idxs_flip] = T_f.vflip(y[idxs_flip])
        y = align_reconstructions.rotate_batch(y, angles=align_transforms[:,0])

    grid = grid_from_2cols(x,y, nrow=nrow, ncol=ncol)
    return grid
