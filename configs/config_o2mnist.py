"""
Config file interpreted by `run.py`
"""
import os
from pathlib import Path

import torchvision.transforms as T

from utils import utils


dir_this = os.path.dirname(os.path.realpath(__file__))  # path of this file
dir_parent = Path(dir_this).parent

config = dict(
    do_logging=True,  # required for model saving, and metric saving to a local directory. Directory is printed to screen.
    data=dict(
        # directory containing data `X_train.sav` and (optionally) `X_test.sav`,`y_train.sav`,`y_test.sav`
        data_dir=os.path.join(dir_parent, "data", "o2_mnist"),
        # number of images per batch in torch DataLoader
        batch_size=256,
        # number of workers in torch DataLoader
        num_workers=0,
        shuffle_data_loader=False,
        # transforms: torchvision-style transforms on train and test (or `None`)
        transform_train=T.Compose(
            (
                T.RandomRotation(180),
                T.RandomVerticalFlip(0.5),
            )
        ),
        transform_test=None,
    ),
    run=dict(
        # how many iterations through the dataset
        epochs=51,
        # whether to test the validation data during training
        do_validation=True,
        # how frequently to run validation code (ignored if do_validation=False)
        valid_freq=10,
    ),
    # model architecture
    model=dict(
<<<<<<< HEAD
        name="vae",       # 'vae' is the only option now
        zdim=128,         # vae bottleneck layer
        channels=1,       # img channels, e.g. 1 for grayscale, 3 for rgb 
        do_sigmoid=True,  # whether to make the output be between [0,1]. Usually True. 
        vanilla=False,    # Regular (vanilla) vae instaed of O2-VAE. If true then set config.model.encoder='cnn' and `config.loss.align_loss=False`
=======
        name="vae",  # 'vae' is the only option now
        zdim=256,  # vae bottleneck layer
        channels=1,  # img channels, e.g. 1 for grayscale, 3 for rgb
        do_sigmoid=True,  # whether to make the output be between [0,1]. Usually True.
        vanilla=False,  # Regular (vanilla) vae instaed of O2-VAE. If true then set config.model.encoder='cnn' and `config.loss.align_loss=False`
>>>>>>> f646eb5c4d9bd683d57303d59635502adf379cbd
        encoder=dict(
            # `name`: 'o2_cnn' for o2-invariant encoder. 'cnn_encoder' for standard cnn encoder.
            name="o2_cnn_encoder",
            # `cnn_dims`: must be 6 elements long. Increase numbers for larger model capacity
            cnn_dims=[6, 9, 12, 12, 19, 25],
            # `layer_type`: type of cnn layer (following e2cnn library examples)
            layer_type="inducedgated_norm",  # recommend not changing
            # `N`: Ignored if `name!='o2'`. Negative means the model will be O2-invariant.
            #     Again, see (e2cnn library examples). Recommend not changing.
            N=-3,
        ),
        decoder=dict(
            name="cnn_decoder",  # 'cnn' is the ony option
            # `cnn_dims`: each extra layer doubles the dimension (image width) by a factor of 2.
            #    E.g. if there are 6 elements, image width is 2^6=64
            cnn_dims=[192, 96, 96, 48, 48],
            out_channels=1,
        ),
    ),
    loss=dict(
        # 'beta' from beta-vae, or the weight on the KL-divergence term https://openreview.net/forum?id=Sy2fzU9gl
        beta=0.01,
        # `recon_loss_type`: "bce" (binary cross entropy) or "mse" (mean square error)
        #    or "ce" (cross-entropy, but warning, not been tested well)
        recon_loss_type="bce",
        # for reconstrutcion loss, pixel mask. Must be either `None` or an array with same dimension as the images.
        mask=None,
        align_loss=True,  # whether to align the output image to the input image
        # whether to use efficient Foureier-based loss alignment. (Ignored if align_loss==False)
        align_fourier=True,
        # whether to do align the best rotation AND flip, instead of just rotation. (Ignored if align_loss==False)
        do_flip=True,
        # if doing brute force align loss, this is the rotation discretization. (Ignored if
        #   align_loss==False or if align_fourier==True)
        rot_steps=2,
        # Recommend not changing. The vae prior distribution. Optoins: ("standard","normal","gmm"). See models.vae.VAE for deatils.
        prior_kwargs=dict(
            prior="standard",
        ),
    ),
    optimizer=dict(
        lr=1e-3,
    ),
    wandb_log_settings=dict(
        # `wandb_anonymous` if you have a wandb logging acocunt, you can log into it and set this to False.
        #    otherwise leave it
        wandb_anonymous=True,
        # `wandb_enbable_cloud_logging`: logging is done to a local folder. To link to online wandb loggin
        #    service, set this to True.
        wandb_enbable_cloud_logging=False,
        # the next 3 are for organizing wandb runs. They can be ignored if not using wandb dashboard.
        project="demo",
        name=None,  # (string) name for this run. If left as None, it will be autogenerated
        group="0",  # wandb logging group (helps organizing many experiments on wandb dashboard)
    ),
    logging=dict(
        train_batch_freq=20,  # how many batches between wandb updates
        tags=[],
        do_progress_bar=True,  # wheter to show the progress bar when running scripts
        do_checkpoint=True,  # whether to save the model
        checkpoint_epoch=10,  # how frequently (in epochs) to save the model (Ignored if do_checkpoint==False)
        print_verbose=1,  # whether to print logging INFO to stdout.
    ),
)
config = utils.Bunch.from_nested_dicts(
    config
)  # this data structure lets you index it like: `config.data.name`
