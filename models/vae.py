import importlib

import numpy as np
import torch
import torchvision

from models import align_reconstructions

from . import model_utils as mut


importlib.reload(mut)


class VAE(torch.nn.Module):
    def __init__(
        self,
        q_net,
        p_net,
        zdim,
        do_sigmoid=True,
        loss_kwargs=dict(
            recon_loss_type="bce",
            beta=0.01,
            align_loss=True,
            align_fourier=True,
            do_flip=True,
            rot_steps=2,
        ),
        prior_kwargs=dict(prior="standard"),
        mask=None,
        **kwargs,
    ):
        """
        Args:
        rot_steps: discretization of the loss function rotations.
        vanilla_vae (bool): if True, then just do the standard loss, not the
            invariant loss.

        Args
            q_net (torch.nn.Module): encoder model.
            p_net (torch.nn.Module): decoder model.
            zdim (int): number of dimensions of the bottleneck/representation layer
            do_sigmoid (bool): whether to do sigmoid on the final layer outputs.
                Default True, but may set to False if doing `binary_cross_entropy_with_logits`
                loss (rather than binary_cross_entropy`. If False, then images
                reconstructed image should be passed through sigmoid manually to
                get image format.

        """
        super().__init__()
        # assign encoder and decoder and count their params
        self.q_net = q_net
        self.p_net = p_net
        self.loss_kwargs = loss_kwargs

        self.zdim = zdim
        self.do_sigmoid = do_sigmoid

        # loss mask if supplied
        self.mask = mut.build_circle_mask(m, n) if mask else None

        # prior setup
        self.prior_setup(prior_kwargs)

        # compute model size and save params to p_params and q_params
        _ = self.model_size()

    def prior_setup(self, prior_kwargs):
        """
        set up hyperparamaters for the prior based on `prior_kwargs` arg
        `standard`: standard normal N(0,I)
        `normal`: normal N(mu, sigma), with learnable mu and sigma.
        `gmm`: mixture of gaussians sum_i [pi_i* N(mu_i, sigma_i)] with learnable
            pi_i, mu_i, sigma_i.
        """
        self.prior_kwargs = prior_kwargs
        self.prior = self.prior_kwargs["prior"]

        valid_priors = ["standard", "normal", "gmm"]
        assert self.prior in valid_priors, f"{self.prior} must be one of {valid_priors}"
        if self.prior == "standard":
            # standard normal N(0,I)
            self.pm = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.pv = torch.nn.Parameter(torch.ones(1), requires_grad=False)

        elif self.prior == "normal":
            # Normal with learnable mean and standard deviation (but no covariance), N(\mu, \sigma^2 I)
            self.pm = (
                torch.nn.Parameter(torch.zeros(1), requires_grad=False)
                + prior_kwargs["pm"]
            )
            self.pv = (
                torch.nn.Parameter(torch.ones(1), requires_grad=False)
                + prior_kwargs["pv"]
            )

        # mixture of gaussias, with learnable gaussian params, but fixed uniform weigths, k
        elif self.prior == "gmm":
            # Gaussian mixture prior
            assert "k" in prior_kwargs.keys(), "prior 'gmm' needs 'k' "
            self.k = prior_kwargs["k"]
            # gaussian params
            scaler = np.sqrt(self.k * self.zdim)
            self.z_pre = torch.nn.Parameter(
                torch.randn(1, 2 * self.k, self.zdim) / scaler
            )

            # Uniform weighting, not learnable
            self.pi = torch.nn.Parameter(torch.ones(self.k) / self.k, requires_grad=False)

        return

    def model_size(self):
        self.q_params = sum([p.numel() for p in self.q_net.parameters()])
        self.p_params = sum([p.numel() for p in self.p_net.parameters()])
        self.params = self.q_params + self.p_params
        repr = f"Parameter counts: \n encoder {self.q_params:,}\n decoder {self.p_params:,}\n"
        repr += f" total   {self.q_params+self.p_params:,}"
        return repr

    def model_details(self):
        repr = f"Encoder type {type(self.q_net)}\nDecoder type {type(self.p_net)}"
        repr += f"\nBottleneck dims {self.zdim}"
        repr += f"\nLoss func: {self.loss_kwargs['recon_loss_type']}, beta={self.loss_kwargs['beta']}"
        repr += f" loss aligns output: {self.loss_kwargs['align_loss']} "
        repr += "\n" + self.model_size()

        return repr

    def sample(self, mu, logvar):
        return mut.sample_gaussian(mu, logvar)

    def embed_data(self, x):
        """Get the learned data embedding from a trained model. Intded to use
        one batch at a time"""
        if len(x) > 256:
            msg = (
                "This function is for batches only. If you need embeddings for more"
                "than 256 imgs, use `utils.utils.get_model_embeddings` instead"
            )
            raise ValueError(
                "This function is for smal batches only. "
                "Data passed to `embed_data` too big. Use"
            )
        self.eval()
        with torch.no_grad():
            z_mu, _ = torch.split(self.q_net(x), self.zdim, dim=1)
        return z_mu

    def encode(self, x):
        z = self.q_net(x)
        mu, logvar = torch.split(z, self.zdim, dim=1)
        return mu, logvar

    def loss_recon(self, x, y):
        """
        Reconstruction using loss type self.loss_kwargs['recon_loss_type'] which
        by default is binary cross entropy. Computes the loss pixel-wise, then
        sums over the image. Returns one number per image in the batch.
        Args:
            x: vae image input shape (bs,c,h,w)
            y: vae reconstruction image shape (bs,c,h,w)
        Returns:
            loss_recon: total loss, shape (bs, )
        """
        if self.mask:
            self.mask = self.mask.to(x.device)

        # case 1: o2-vae - align the input and output images
        if self.loss_kwargs.get("align_loss", False):
            # case 1: efficient alignment using Fourier space methods
            if self.loss_kwargs["align_fourier"]:
                loss_recon, _ = align_reconstructions.loss_reconstruction_fourier_batch(
                    x,
                    y,
                    recon_loss_type=self.loss_kwargs["recon_loss_type"],
                    mask=self.mask,
                )
            # case 2: manually check a discretised set of rotations: more computation time+memory
            else:
                # get the min loss of all the rotations/flips per image
                loss_recon, _ = align_reconstructions.loss_smallest_over_rotations(
                    x,
                    y,
                    rot_steps=self.loss_kwargs.get("rot_steps", 2),
                    do_flip=self.loss_kwargs.get("do_flip", True),
                    recon_loss_type=self.loss_kwargs["recon_loss_type"],
                    return_details=0,
                    mask=self.mask,
                )

        # case 2: conventional loss without aligning input and output
        else:
            # standard loss term / not invariant or anything.
            x, y = x.view(len(x), -1), y.view(len(x), -1)
            if self.loss_kwargs["recon_loss_type"] == "bce":
                loss_func = torch.nn.functional.binary_cross_entropy
            elif self.loss_kwargs["recon_loss_type"] == "mse":
                loss_func = torch.nn.functional.mse_loss
            else:
                raise ValueError()
            loss_recon = loss_func(y, x, reduction="none").sum(1)

        return loss_recon

    def loss_kl(self, mu, logvar):
        """
        KL divergence per element in the batch.
        Depending on the method, we use either inferred params or a sampled z.
        """
        if self.prior in ("standard", "normal"):
            return mut.kl_normal(mu, torch.exp(logvar), self.pm, self.pv)

        elif self.prior in ("gmm"):
            # this repeats the z_sample operation a little unnecessarily, so this
            # isn't ideal
            z_sample = self.sample(mu, logvar)
            bs = len(z_sample)

            # First term of KL monte carlo estimator is likelihood of the sample
            log_prob_q_z_x_batch = mut.log_normal(z_sample, mu, torch.exp(logvar))

            # get the current state of the prior and duplicate over the batch
            pm, pv = mut.gaussian_parameters(self.z_pre, dim=1)
            pm, pv = mut.duplicate(pm, bs), mut.duplicate(pv, bs)

            # Using the prior, and the sample, comput the log-probability of z under the gaussian mixture
            log_prob_gauss_mixture_batch = mut.log_normal_mixture(z_sample, pm, pv)

            # combine the terms to get the KL divergence as a batch
            kl_batch = log_prob_q_z_x_batch - log_prob_gauss_mixture_batch

            return kl_batch
        else:
            raise ValueError(f"No implementation for prior type {self.prior}")

    def loss(self, x, y, mu, logvar):
        """
        Returns sample-average loss for the batch.
        """
        loss_recon = self.loss_recon(x, y).mean()
        loss_kl = self.loss_kl(mu, logvar).mean()
        loss = loss_recon + self.loss_kwargs["beta"] * loss_kl
        return dict(
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
            beta=self.loss_kwargs["beta"],
        )

    def reconstruct(self, x, reorient=0, dummy=None):
        _, y, _, _ = self.forward(x)
        # multiclass classification output
        if self.loss_kwargs.get("recon_loss_type", "bce") == "ce":
            y = torch.nn.Softmax2d()(y)
            y = y.argmax(dim=1, keepdim=True)
            y = y / y.max()
        return y

    def decode_img(self, z):
        """
        Decode the image from a latent point. This may need to be different from
        calling p_net directly in case there is a sigmoid call req'd.
        """
        y = self.p_net(z)
        if self.do_sigmoid:
            y = torch.sigmoid(y)
        if self.recon_loss_type == "ce":
            y = torch.nn.Softmax2d()(y)
            y = xhat.argmax(dim=1, keepdim=True)
            y = y / y.max()

        return y

    def forward(self, x):
        in_shape = x.shape
        bs = in_shape[0]
        assert x.ndim == 4

        # inference and sample
        z = self.q_net(x)
        mu, logvar = torch.split(z, self.zdim, dim=1)
        z_sample = self.sample(mu, logvar)
        assert mu.shape == logvar.shape

        # decode
        y = self.p_net(z_sample)
        if self.do_sigmoid:
            y = torch.sigmoid(y)
        # check the spatial dimensions are good (if doing multiclass prediction per pixel, the `c` dim may be different)
        assert in_shape[-2:] == y.shape[-2:], (
            "output image different dimension to "
            "input image ... probably change the number of layers (cnn_dims) in the decoder"
        )

        return x, y, mu, logvar
