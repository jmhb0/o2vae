mport numpy as np
import skimage
import torch
import torchvision.transforms.functional as TF
from skimage import measure
import tqdm

def align_major_axis(data, align_axis=0):
    """ 
    For each element of `data` shape (n,1,y,x), orient the object so its longest
    axis is aligned to `y`. Here `longest` is the definition of skimage.measure.regionprops
    `orientation` value which is a first order ellipse approximation. 
    """
    data_out = torch.zeros_like(data, dtype=torch.float32)
    props = []
    for i in range(len(data)):
        img = data[i]
        import matplotlib.pyplot as plt

        r = skimage.measure.regionprops((img[align_axis]+0.9999).numpy().astype(int))[0]
        props.append([
            r.orientation, r.axis_major_length, r.axis_minor_length,
        ])
        img_rot = TF.rotate(img, angle=-r.orientation*180/torch.pi)
        data_out[i] = img_rot
    return data_out.float()

def get_mean_cell(data):
    """ 
    For shape n,c,y,x, get the mean of pixels over the n datapoints.
    The output is shape (1,c,y,x)
    """
    n, c, ydim, xdim = data.shape
    # assert c==1
    mean_cell = np.reshape(
        np.mean(np.reshape(data, (n,c*ydim*xdim)), 0),
        (1, c, ydim, xdim)
    )
    return mean_cell

def compute_cell_to_mean_cell_err(data_flip, mean_cell):
    """
    For dataset size (n,1,y,x), compute the per-pixel error compared to the 
    mean cell with shape (1,1,y,x), then average over the y,x dimension to get 
    a mean error for each n. The output is shape (n,).
    """
    err = (data_flip - mean_cell)**2
    err = np.mean(np.reshape(err, (len(err),-1)), 1)**0.5
    return err

def compute_cell_batch_err(x, y):
    """ for x and y both size (b,c,x,y)
    """
    # the mean csell function works for this too but I already named it too specifically
    # so I will just call it.
    return compute_cell_to_mean_cell_err(x,y)

def align_refinement_flips(data_aligned, n_iters=20, do_plot=0, print_err=1, align_axes=[0], verbose=1):
    """
    Do the alignment refinement procedure described in `Comparison of quantitative 
    methods for cell-shape analysis`, in the paragraph starting with the line 
        > "Because there were shape symmetries not captured ... "
    The input must already have been aligned by some basic procedure, probably 
    of aligning the longest axis to the x- or y- using PCA or the first order 
    ellipse. That is, it should be aligned up to some reflection. 
    
    Args
        align_axes (lst): the axes that will be used for compute the alignment. E.g. you might 
            have a 2-channel image and you could align using only axis [0] or both [0,1]. 
    """
    if verbose:
        print(f"align axes", align_axes)
    assert type(align_axes) is list 

    data = np.array(data_aligned)
    n=len(data)

    initial_mean_cell = get_mean_cell(data)
    for i in range(n_iters):
        ## compute the mean cell
        mean_cell = get_mean_cell(data[:,align_axes])

        ## compute the 4 flip permutations for each image
        data_flip00 = data.copy()                  # no flip
        data_flip10 = np.flip(data, axis=2).copy() # flip y
        data_flip01 = np.flip(data, axis=3).copy() # flip x
        data_flip11 = np.flip(np.flip(data, axis=3), axis=2).copy() # flip x and y

        if do_plot:
            f, axs = plt.subplots(1,4)
            axs[0].imshow(data[0,0])
            axs[1].imshow(data[1,0])
            axs[2].imshow(data[2,0])
            axs[3].imshow(data[3,0])

        ## compute the cross RMSE for each per data point
        err00 = compute_cell_to_mean_cell_err(data_flip00[:,align_axes], mean_cell)
        err10 = compute_cell_to_mean_cell_err(data_flip10[:,align_axes], mean_cell)
        err01 = compute_cell_to_mean_cell_err(data_flip01[:,align_axes], mean_cell)
        err11 = compute_cell_to_mean_cell_err(data_flip11[:,align_axes], mean_cell)

        current_error = err00.mean()
        if print_err:
            print(f"Iter {i} error {current_error:.6f}")

        # find the min error permutation
        errs = np.stack((err00, err10, err01, err11),1)
        # err00.mean(), err10.mean(), err01.mean(), err11.mean()
        min_flip_choice = np.argmin(errs, 1)

        # replace data where appropriate
        idxs_00 = np.where(min_flip_choice==0)[0]
        data[idxs_00] = data_flip00[idxs_00].copy() # actually does nothing

        idxs_01 = np.where(min_flip_choice==1)[0]
        data[idxs_01] = data_flip10[idxs_01].copy()

        idxs_10 = np.where(min_flip_choice==2)[0]
        data[idxs_10] = data_flip01[idxs_10].copy()

        idxs_11 = np.where(min_flip_choice==3)[0]
        data[idxs_11] = data_flip11[idxs_11].copy()

    final_mean_cell = mean_cell
    data_flip_aligned = data.copy()
    
    return data_flip_aligned, initial_mean_cell, final_mean_cell, current_error

