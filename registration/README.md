# Efficient image registration: module for rotation and flip image alginment on GPUs

In image registration, we want to find the flip and rotation that best aligns one image with a reference image (in terms of cross-correlation). An efficient Fourier-based method was proposed in  ["An FFT-based technique for translation, rotation, and scale-invariant image registration"](https://ieeexplore.ieee.org/abstract/document/506761). 

In `./registration.py`, we implement this approach for GPUs using PyTorch functions. This means we can register a large batch of image pairs at the same time, thus saving computational resources. (Our model needs to do this for every batch during training, so the saving is significant). 

There are two steps
- Map image pairs to polar coordinates. 
- Apply phase correlation analysis to recover the correct rotation.

# Usage:
We have two Pytorch tensors of images `x` and `y` with shape `(n_images,n_channels,height,width)`, and each `x[i]` must be registered against `y[i]`.
```
import registration

# put the tensors on gpu if it's available. 
device='cuda'  
x, y = x.to(device), y.to(device)

# object for doing polar coordinate transform
PolarTransform = registration.PolarTranformBatch(image_shape=x.shape[-2:])
# map to polar coordinates, x, y
x_polar, y_polar = PolarTransform.warp_batch(x), PolarTransform.warp_batch(y)

# recover the best rotations using polar coordinates 
shifts, error, phasediff = registration.phase_correlation_2d_batch(x_polar, y_polar)
angle_r, scale = -shifts_r[:,0]
```

`angle_r` and `scale` are both arrays with shape `n_examples,` having the optimal rotation and scaling for each pair. To compute the optimal flip, simply repeat the analysis on a flipped version of `y` and iff the `error` variable is smaller, then choose the flipped version. 

Note: These functions are similar to scikit image `transform.warp_polar` and `registration.phase_cross_correlation`

# Future extension:
- Currently we only solve for rotation and scaling. We will add translation. 
