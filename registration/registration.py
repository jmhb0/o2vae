"""
Code for the 'registration' module. Given two input images, find the rotation and 
flip that optimally aligns them in terms of image cross-correlation.

To use the efficient version (item 1 in the above list) you first need to transform
the image to polar coords. 

More details, see 
    https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html
"""
import numpy as np
import skimage
import torch
import torchgeometry as tgm
import torchvision.transforms.functional as T_f


class PolarTranformBatch:
    """
    Take the polar transform (or log-polar transform) of a batch of pytorch images
    efficiently. This is mostly the same as skimage.transform.warp_polar except its
    done batch-wise and also with less erorr handling, and fewer features.
    This exists as a class because we want to build a coordinate
    mapping matrix once, and then use it many times during a training epoch.
    """

    def __init__(
        self,
        image_shape,
        center=None,
        radius=None,
        output_shape=None,
        scaling="linear",
        order=1,
        cval=0,
        mode_grid_sample_padding="zeros",
        mode_grid_sample_interpolate="bilinear",
        mode_padding_ndimage="constant",
    ):
        """
        order (int): interpolation order 0 nearest, 1 bilinear, 3 quadratic.
        cval (float): pad value

        mode_grid_sample_interpolate: how to interpolate between mapped image points in the mapped grid,
            see torch.nn.functional.grid_sample, param `mode`
        mode_padding_ndimage: how to deal with boundaries for the `warp_single_ndimage` function, which is not
            the batch version that we use. See `scipy.ndimage.map_coordinates`.
        """
        assert len(image_shape) == 2
        self.order = order
        self.cval = cval
        self.mode_grid_sample_padding = mode_grid_sample_padding
        self.mode_grid_sample_interpolate = mode_grid_sample_interpolate
        self.mode_padding_ndimage = mode_padding_ndimage

        ydim, xdim = image_shape

        if center is None:
            center = (torch.Tensor([ydim, xdim]) / 2) - 0.5

        if radius is None:
            w, h = ydim / 2, xdim / 2
            radius = (w**2 + h**2) ** 0.5

        if output_shape is None:
            height = 360
            width = int(np.ceil(radius))
            output_shape = (height, width)
        else:
            output_shape = safe_as_int(output_shape)
            height = output_shape[0]
            width = output_shape[1]

        if scaling == "linear":
            k_radius = width / radius
            self.map_func = skimage.transform._warps._linear_polar_mapping
        elif scaling == "log":
            k_radius = width / np.log(radius)
            self.map_func = skimage.transform._warps._log_polar_mapping
        else:
            raise ValueError("Scaling value must be in {'linear', 'log'}")

        k_angle = height / (2 * torch.pi)
        self.warp_args = {
            "k_angle": k_angle,
            "k_radius": k_radius,
            "center": center.numpy(),
        }

        inverse_map = self.map_func

        def coord_map(*args):
            return inverse_map(*args, **self.warp_args)

        # coordinate map for warping in ndimage format
        self.coords = torch.Tensor(
            skimage.transform._warps.warp_coords(coord_map, output_shape)
        )

        # the same coordinate map but for torch: the order of channels is different and images
        # are normalized to the range [-1,1] to suit the function `torch.nn.functional.grid_sample`
        self.coords_torch_format = self.coords.clone().moveaxis(
            0, 2
        )  # move to shape (2,Y,X) to (Y,X,2)
        min0, min1, max0, max1 = 0, 0, image_shape[0], image_shape[1]
        self.coords_torch_format[:, :, 0] = (self.coords_torch_format[:, :, 0] - min0) / (
            max0 - min0
        ) * 2 - 1
        self.coords_torch_format[:, :, 1] = (self.coords_torch_format[:, :, 1] - min1) / (
            max1 - min1
        ) * 2 - 1
        self.coords_torch_format = self.coords_torch_format[
            :, :, [1, 0]
        ]  # grid_sample has the order reversed

        # Pre-filtering not necessary for order 0, 1 interpolation
        self.prefilter = order > 1
        self.ndi_mode = skimage._shared.utils._to_ndimage_mode(self.mode_padding_ndimage)

    def warp_single_ndimage(self, x):
        """
        Test polar transform of a single image using the ndimage library.
        This does not make use of any GPU batch stuff (you can only
        transform one image at a time), so it is a baseline for how this function
        should work.
        """
        assert x.ndim == 2
        if type(x) is torch.Tensor:
            x = x.cpu().numpy()
        warped = ndi.map_coordinates(
            x,
            self.coords,
            prefilter=self.prefilter,
            mode=self.ndi_mode,
            order=self.order,
            cval=self.cval,
        )
        return torch.Tensor(warped)

    def warp_batch(self, x):
        """
        Do polar transform as a batch, which should be fast if using a GPU.
        """
        assert x.ndim == 4
        batch_size = len(x)
        coords_batch = (
            self.coords_torch_format.unsqueeze(0)
            .expand(batch_size, *self.coords_torch_format.shape)
            .to(x.device)
        )
        warped_batch = torch.nn.functional.grid_sample(
            x,
            coords_batch,
            align_corners=True,
            padding_mode=self.mode_grid_sample_padding,
            mode=self.mode_grid_sample_interpolate,
        )
        return warped_batch

    def _verify_warping_funcs_are_the_same(self, x=None, do_plot=1):
        """
        Take `x` with shape (Y,X) if supplied, or on skimage.data.checkboard if `x` is None.
        Run warp_single_ndimage and warp_batch on it. The point is that it
        should be the same.
        """
        if x is None:
            import skimage

            print("No image data provided, using skimage.data.checkerboard()")
            x = torch.Tensor(skimage.data.checkerboard())  # a test image
        print(x.shape, len(x.shape), x.ndim)
        warped_ndimage = self.warp_single_ndimage(x)
        x_batch = (
            x.unsqueeze(0).unsqueeze(0).expand(1, 1, *x.shape)
        )  # put the test image into a batch

        warped_batch = self.warp_batch(x_batch)

        if do_plot:
            f, axs = plt.subplots(1, 2)
            axs[0].imshow(warped_ndimage)
            axs[1].imshow(warped_batch[0, 0])
            return warped_ndimage, warped_batch[0, 0], f
        else:
            return warped_ndimage, warped_batch[0, 0], None

    def _test_runtime(self, n=50, bs=256, ysz=128, xsz=128):
        """
        Simulate running the polar transform n times with batch size bs.
        Do it for the batch version, then the single image version, and
        then batch version on cuda if cuda is available.

        E.g. with defaults, expect the batch version to be 10x faster.
        For a fixed amount of data (the same value n*bs), the batch version
        will be a faster with bigger `bs` and smaller `n`, but the serial
        version should be the same.
        """
        import time

        import skimage
        import torchvision.transforms.functional as T_f

        image = torch.Tensor(data.checkerboard())  # a test image
        image_batch = image.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
        image_batch = T_f.resize(image_batch, (ysz, xsz))

        print(f"Testing {n} iterations, batch size {bs}, image dimension ({ysz},{xsz})")
        print(image_batch.shape)

        start = time.time()
        for i in range(n):
            _ = self.warp_batch(image_batch)
        print(f"{time.time()-start:.2f} sec for batch version")

        start = time.time()
        for i in range(n):
            for img in image_batch:
                _ = self.warp_single_ndimage(img[0])
        print(f"{time.time()-start:.2f} sec for serial version")

        if torch.cuda.is_available():
            start = time.time()
            for i in range(n):
                image_batch_ = image_batch.cuda()
                _ = self.warp_batch(image_batch_)
            print(f"{time.time()-start:.2f} sec for cuda batch version")


def _torch_unravel_index_batch(index, shape):
    """
    Modified slighly from
        https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode="trunc")  ## equiv to index // dim
    return tuple(reversed(out))


def _compute_error(CCmax, src_amp, target_amp):
    error = 1 - (CCmax * CCmax.conj()) / (src_amp * target_amp)
    return torch.abs(error) ** 0.5


def _compute_phasediff(CCmax):
    return torch.atan2(CCmax.imag, CCmax.real)


def phase_correlation_2d_batch(
    x, y, normalization=None, upsample_factor=1, return_error=1, space="real"
):
    """
    upsample_factor: NotImplemented for anything other than 1.
    """
    device = x.device
    reference_image, moving_image = x, y
    assert reference_image.ndim == reference_image.ndim == 4, "must be shape (bs,1,Y,X)"
    bs, c, ydim, xdim = reference_image.shape
    assert c == 1, "input must be grayscale, having shape (bs,1,Y,X)"

    # take ffts and cross correlations
    src_freq = torch.fft.fft2(x)
    target_freq = torch.fft.fft2(y)
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if normalization == "phase":
        eps = torch.finfo(image_product.real.dtype).eps
        image_product /= np.maximum(np.abs(image_product), 100 * eps)
    elif normalization is not None:
        raise ValueError()
    cross_correlation = torch.fft.ifft2(image_product)

    # error stuf
    if return_error:
        src_amp = torch.real(src_freq * src_freq.conj()).view(bs, -1).sum(1)
        target_amp = torch.real(target_freq * target_freq.conj()).view(bs, -1).sum(1)

    # find the max correlation
    cross_correlation_abs = torch.abs(cross_correlation)
    cross_correlation_abs = cross_correlation_abs.view(bs, ydim, xdim)

    # ids and values of the peak of the cross corr thing
    max_idxs_batch = torch.argmax(cross_correlation_abs.view(bs, ydim * xdim), 1)
    CCmax = cross_correlation.view(bs, -1)[np.arange(bs), max_idxs_batch]
    ymax_batch, xmax_batch = _torch_unravel_index_batch(max_idxs_batch, (ydim, xdim))

    # work out the shift - case depends on whether it's above or below the midpoint
    midpoints = torch.Tensor([int(axis_size / 2) for axis_size in shape[-2:]]).to(device)
    shifts = torch.stack((ymax_batch, xmax_batch), 1).to(device).float()
    shifts[shifts > midpoints] -= (
        torch.Tensor([[ydim, xdim]])
        .to(device)
        .float()
        .expand((bs, 2))[shifts > midpoints]
    )

    if upsample_factor == 1:
        pass
    else:
        raise NotImplementedError()

    return shifts, _compute_error(CCmax, src_amp, target_amp), _compute_phasediff(CCmax)
