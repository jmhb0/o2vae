import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as T_f
<<<<<<< HEAD
import torchgeometry as tgm
import numpy as np
=======
from torchvision.utils import make_grid
>>>>>>> f646eb5c4d9bd683d57303d59635502adf379cbd

from models import align_reconstructions


def grid_from_2cols(
    x1,
    x2,
    nrow=10,
    ncol=8,
    grid_kwargs={"padding": 2, "pad_value": 0.5, "align_corners": False},
):
    """
    Given 2 tensors of images x1, x2, put them in a single imagegrid (by
    calling torchvision.utils.make_grid) where x1[i] is next to x2[i].
    It's useful for plotting an original image and its reconstruction.
    """
    n = nrow * (ncol // 2)
    x1, x2 = x1[:n], x2[:n]
    # add gray bars to separate the columns
    gray_bar = 0.2 * torch.ones((*x1.shape[:3], 2))
    x1 = torch.cat((gray_bar, x1), dim=3)
    x2 = torch.cat((x2, gray_bar), dim=3)

    # merge the images and create a grid
    assert x1.shape == x2.shape
    x = torch.zeros((len(x1) * 2, *x1.shape[1:]), dtype=x1.dtype)
    x[0::2] = x1
    x[1::2] = x2
    grid = make_grid(x, ncol, **grid_kwargs)
    grid = torch.permute(grid, (1, 2, 0))
    return grid


def grid_from_3cols(x1, x2, x3, nrow=10, ncol=6):
    """
    Same as `grid_from_2cols` but for 3 columns.
    """
    n = nrow * (ncol // 3)
    x1, x2, x3 = x1[:n], x2[:n], x3[:n]

    assert x1.shape == x2.shape
    x = torch.zeros((len(x1) * 3, *x1.shape[1:]), dtype=x1.dtype)
    x[0::3] = x1
    x[1::3] = x2
    x[2::3] = x3
    grid = make_grid(x, ncol)
    grid = torch.permute(grid, (1, 2, 0))
    return


def reconstruction_grid(model, x, align=True, nrow=12, ncol=8, device="cuda"):
    """
    Run on cpu, and non-efficient loops, because we assume it's  not run too often.
    Given a trained model and batch of data with shape (N,C,Y,X), return a tensor
    that can be plotted as an image using matplotlib imshow().
    grid that can
    where one column is
    """
    assert len(x) <= 256
    model.eval().to(device)
    with torch.no_grad():
        y = model.reconstruct(x.to(device)).cpu()

    # rotate and flip y to align with x
    if align:
        # array `align_transforms` is the optimal rotation and flip
        _, align_transforms = align_reconstructions.loss_reconstruction_fourier_batch(
            x, y, recon_loss_type=model.loss_kwargs["recon_loss_type"]
        )
        # flip and rotate
        idxs_flip = np.where(align_transforms[:, 1])[0]
        y[idxs_flip] = T_f.vflip(y[idxs_flip])
        y = align_reconstructions.rotate_batch(y, angles=align_transforms[:, 0])

    grid = grid_from_2cols(x, y, nrow=nrow, ncol=ncol)
    return grid

def rotate_batch(x, angles):
    """
    Rotate many images by different angles in a bathch
    Args
        x (torch.Tensor): image batch shape (bs, c, y, x)
        angles (torch.Tensor): shape (bs,) list of angles to rotate `x`
    """
    assert len(x)==len(angles)
    assert x.ndim==4
    bs = len(x)
    h, w = x.shape[-2:]
    center = torch.Tensor([[h,w]]).expand(bs,2) *0#/ 2 + 0.5
    scale = torch.ones((bs))
    M = tgm.get_rotation_matrix2d(center, -angles, scale)
    grid = torch.nn.functional.affine_grid(M, size=x.shape).to(x.device)
    rotated = torch.nn.functional.grid_sample(x, grid)

    return rotated

def rotated_flipped_xs(x, rot_steps, trans=None, do_flip=True, upsample=0):
    """
    x: batch of images (b,c,h,w)
    rot_steps: number of rotations to try.
    trans: the list of transformatio ops to apply. The caller may want to precompute
        this if calling it repeatedly.
    upsample: If 0 then do nothing. If>0, then upsample by this factor to the
        original image before performing the transformation, then downsample again after.
    Returns (len(rots),b,c,h,w)
    """
    bs, c, h,w = x.shape
    angles = torch.arange(0,360, rot_steps)
    n_angles = len(angles)
    n_permutations = n_angles*(1+do_flip)
    # new tensor to hold a permuted (rotated and possibly flipped) versions, size (n_permutaitons,bs,c,y,x)
    xs = torch.zeros((n_permutations, bs, c,h,w), device=x.device)

    # copy the original angle n_angles time in the 1st dimension, then flatten into one big batch
    x_expanded = x.unsqueeze(0).expand(n_angles, *x.shape).contiguous().view(n_angles*bs,c,h,w)
    # copy the angles dimension the same number of times
    angles = angles.unsqueeze(1).expand(n_angles, bs).flatten()
    x_expanded = rotate_batch(x_expanded, angles)
    xs[:n_angles] = x_expanded.view(n_angles, bs, c,h,w).clone()
    # if doing vflip, then put that in as well
    if do_flip:
        xs[n_angles:] = T_f.vflip(x_expanded).view(n_angles, bs, c,h,w)
    return xs
