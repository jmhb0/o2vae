"""
"""
from registration import registration
import torch
import numpy as np
import torchvision.transforms.functional as T_f
import torchgeometry as tgm
import skimage

def loss_reconstruction_fourier_batch(x,y, recon_loss_type='bce', mask=None):
    """
    Compute the O2 reconstruction error between image x[i] and y[i] (batch). 
    That is, find the orientation of y with the lowest registration error with x, 
    according by taking the polar transform with `PolarTranformBatch.warp_batch`
    and correlating with `phase_correlation_2d_batch`. Apply that transformation, 
    then compute the loss. Return the loss and a transformation, `trans_all` array 
    of size (batch,2) which returns the optimal angle and flip used. 
    """
    bs,c,h,w=x.shape
    if recon_loss_type=='bce': loss_func=torch.nn.functional.binary_cross_entropy
    elif recon_loss_type=='mse': loss_func=torch.nn.functional.mse_loss
    else: raise ValueError(f"loss func not supported")

    # get the flip
    yf = T_f.vflip(y)

    # take the polar transform
    PolarTransform = registration.PolarTranformBatch(image_shape=x.shape[-2:], scaling="linear")        
    x_polar, y_polar, yf_polar = PolarTransform.warp_batch(x), PolarTransform.warp_batch(y), PolarTransform.warp_batch(yf)

    # compute the rotation 
    shifts_r, error_r, phasediff_r = registration.phase_correlation_2d_batch(x_polar, y_polar)
    angle_r = -shifts_r[:,0]
    shifts_f, error_f, phasediff_f = registration.phase_correlation_2d_batch(x_polar, yf_polar)
    angle_f = -shifts_f[:,0]

    # apply the rotation to y for both the standard and the flipped versions
    center = torch.zeros(bs,2).to(x.device)
    scale = torch.ones(bs).to(x.device)
    M_r = tgm.get_rotation_matrix2d(center=center, angle=-angle_r, scale=scale).to(x.device)
    M_f = tgm.get_rotation_matrix2d(center=center, angle=-angle_f, scale=scale).to(x.device)
    grid_r = torch.nn.functional.affine_grid(M_r, size=x.shape, align_corners=True).to(x.device)
    grid_f = torch.nn.functional.affine_grid(M_f, size=x.shape, align_corners=True).to(x.device)
    y_out = torch.nn.functional.grid_sample(y, grid_r, align_corners=True)
    yf_out = torch.nn.functional.grid_sample(yf, grid_f, align_corners=True)

    # compute the loss for flip and no flip separately
    loss_r = loss_func(y_out,x, reduction="none", ).view(bs, -1).sum(1)
    loss_f = loss_func(yf_out,x, reduction="none").view(bs, -1).sum(1)
    if mask is not None:
        mask = mask.view(1,-1)
        loss_r, loss_f = loss_r*mask, loss_f*mask
    
    """
    # alternative approach - align the input instead of the output
    xf = T_f.vflip(x)
    # take the polar transform
    PolarTransform = PolarTranformBatch(image_shape=x.shape[-2:], scaling="linear")        
    x_polar, xf_polar, y_polar = PolarTransform.warp_batch(x), PolarTransform.warp_batch(xf), PolarTransform.warp_batch(y)
    # compute the rotation 
    shifts_r, error_r, phasediff_r = phase_correlation_2d_batch(x_polar, y_polar)
    angle_r = -shifts_r[:,0]
    shifts_f, error_f, phasediff_f = phase_correlation_2d_batch(xf_polar, y_polar)
    angle_f = -shifts_f[:,0]
    center = torch.zeros(bs,2).to(x.device)
    scale = torch.ones(bs).to(x.device)
    M_r = tgm.get_rotation_matrix2d(center=center, angle=angle_r, scale=scale).to(x.device)
    M_f = tgm.get_rotation_matrix2d(center=center, angle=angle_f, scale=scale).to(x.device)
    grid_r = torch.nn.functional.affine_grid(M_r, size=x.shape, align_corners=True).to(x.device)
    grid_f = torch.nn.functional.affine_grid(M_f, size=x.shape, align_corners=True).to(x.device)
    x_aligned = torch.nn.functional.grid_sample(x, grid_r, align_corners=True)
    xf_aligned = torch.nn.functional.grid_sample(xf, grid_f, align_corners=True)
    # compute the loss for flip and no flip separately
    loss_r = loss_func(y,x_aligned, reduction="none").view(bs, -1).sum(1)
    loss_f = loss_func(y,xf_aligned, reduction="none").view(bs, -1).sum(1)
    """

    # choose the right loss 
    is_flip = (loss_f < loss_r) 
    loss, angle = loss_r, angle_r
    loss[is_flip], angle[is_flip] = loss_f[is_flip], angle_f[is_flip]
    # save the transformations 
    trans_all=torch.stack((angle, is_flip), 1)

    return loss, trans_all

###############################################################################
# The folowing is the 'brute force' approach that is slower and more memory-
# intensive (scaling with O(bn^2) where n is image width. It therefore requires
# smaller batch sizes, especially for larger imageso
# The relevant function is `loss_smallest_over_rotations`.  

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

def loss_smallest_over_rotations(x, y, rot_steps=1, do_flip=True, mask=None,
        return_details=False, recon_loss_type="bce"):
    """
    For input x, find the rotation of y that has the lowest reconstruction loss
    with x. Return its loss (which also has its gradients attached), the rotation
    that achieves the min loss, and return that image.
    Do it across a batch of images, x.
    Args
    return_details (bool): whether to compute and return the ideal angle
    recon_loss_type (str)" one of "bce" or "ce". ce is the cross-entropy version
     where multiple classes are predicted.  
    Returns:
        loss (tensor): size (bs,)
        theta_min (tensor): size (bs,) how much rotation to apply to y[i] to best
            align it with x[i], in the sense of giving the lowest reconstruction loss.
            If return_details=False then return None
    """
    # get the set of rotated inputs.
    bs, c, m, n = x.shape
    _, c_y, _, _ = y.shape # x annd y may have different shapes if recon_loss_type=='ce' (doing multiclass classification)
    xs = rotated_flipped_xs(x, rot_steps=rot_steps, do_flip=do_flip) # use rot_steps val defined in __init__
    n_rots = len(xs)

    # duplicate the y dimension for each rotation
    y = y.view(1, bs, c_y, m, n).expand(n_rots, bs, c_y, m, n) # duplicate along the first dimension

    # in the first step compute the loss at each pixel. Don't sum over pixels yet so that we can
    # apply the loss mask (if it was supplied)
    if recon_loss_type in ("bce","mse"):
        # flatten the batch dimension and compute cross entropy loss for each pair
        xs = xs.view(n_rots*bs, -1)
        y = y.reshape(n_rots*bs, -1)
        ## here at Jul 13
        if recon_loss_type=="bce":
            loss = torch.nn.functional.binary_cross_entropy(y,xs, reduction="none")
        elif recon_loss_type=="mse":
            loss = torch.nn.functional.mse_loss(y, xs, reduction="none")
        else:
            raise ValueError("code implementation error")

    elif recon_loss_type=="ce":
        xs = xs.swapaxes(2,4).reshape(n_rots*bs*m*n) # the target index
        y = y.swapaxes(2,4).reshape(n_rots*bs*m*n,c_y) # the class-wise logits
        loss = torch.nn.functional.cross_entropy(y,xs.long(), reduction="none")
        loss = loss.view(n_rots*bs, -1)

    # mask out pixels if the mask was supplied
    if mask is not None:
        mask = mask.view(1,-1)
        loss = loss*mask

    # sum over pixels
    loss=loss.sum(1)

    # reshape the loss and the output image back to (n_rots, batch)
    loss = loss.view(n_rots, bs)
    #xs = xs.view(n_rots, bs, c, m, n)
    # y = y.reshape(n_rots, bs, c, m, n)

    # pick out the min loss ad its index
    loss, loss_idx = loss.min(0)

    # if we only care about the loss (e.g. during training), just return the loss
    if not return_details:
        return loss, None
    # otherwise, find the minimal angle (not too stoked about this one)
    else:
        trans = make_trans_list(rot_steps, do_flip=do_flip)
        trans_min = []
        for i in range(len(loss_idx)):
            t = trans[loss_idx[i]]
            if isinstance(t, T.RandomRotation):
                theta=-t.degrees[0]
                do_flip=0
            elif isinstance(t, T.Compose):
                theta = -t.transforms[0].degrees[0]
                assert isinstance(t.transforms[1], T.RandomVerticalFlip)
                do_flip=1
            else:
                raise ValueError()
            trans_min.append((theta, do_flip))

        return loss, trans_min

