import torch
import numpy as np
import os
from data.datasets import TensorDatasetWithTransform
from torch.utils.data import DataLoader
import logging

from models.vae import VAE
from models.encoders_o2.e2scnn import E2SFCNN
from models.encoders_vanilla.cnn_encoder import CnnEncoder
from models.decoders.cnn_decoder import CnnDecoder

import warnings
warnings.filterwarnings('ignore', message='.*aten/src/ATen/native*', ) # filter 2 specific warnings from e2cnn library

def get_datasets_from_config(config):
    assert os.path.isdir(config.data.data_dir), f"config.data.data_dir does not exist"
    data_files = os.listdir(config.data.data_dir)
    assert "X_train.sav" in data_files, f"config.data.data_dir does not contain train data, X_train.sav: {config.data.data_dir}"
    data_x = torch.load(os.path.join(config.data.data_dir, "X_train.sav"))
    data_x = torch.Tensor(data_x)
    if "y_train.sav" in data_files:
        data_y = torch.load(os.path.join(config.data.data_dir, "y_train.sav"))
        if type(data_y) is np.array: data_y=torch.Tensor(data_y)
    else:
        data_y = torch.arange(len(data_x))
    assert len(data_x)==len(data_y)

    dset = TensorDatasetWithTransform(data_x, data_y, transform=config.data.transform_train)
    loader = DataLoader(dset, batch_size=config.data.batch_size, 
        num_workers=config.data.num_workers, shuffle=config.data.shuffle_data_loader)

    if "X_test.sav" in data_files:
        data_x_test = torch.load(os.path.join(config.data.data_dir, "X_test.sav"))
        data_x_test=torch.Tensor(data_x_test)

        if "y_test.sav" in data_files:
            data_y_test = torch.load(os.path.join(config.data.data_dir, "y_test.sav"))
            if type(data_y_test) is np.array: data_y_test=torch.Tensor(data_y_test)

        dset_test = TensorDatasetWithTransform(data_x_test, data_y_test, transform=config.data.transform_train)
        loader_test = DataLoader(dset_test, batch_size=config.data.batch_size, 
            num_workers=config.data.num_workers, shuffle=False)
    else:
        config.run.do_validation=False
        config.logging(f"Did not find 'X_test.sav' and 'y_test.sav' in data_dir. Training will skip validation")
        dset_test, loader_test = None, None

    return dset, loader, dset_test, loader_test

def build_model_from_config(config):
    if config.model.vanilla:
        logging.warning("Using vanila (not O2-invariant) VAE")
        config.model.encoder.name='cnn_encoder'
        config.loss.align_loss = False

    # class lookups for encoder, decoder, and vae
    lookup_model = dict(vae=VAE, o2_cnn_encoder=E2SFCNN, cnn_encoder=CnnEncoder,
            cnn_decoder=CnnDecoder)

    # encoder
    config.model.encoder.n_classes = config.model.zdim*2  # bc vae saves mean and stdDev vecors
    q_net_class=lookup_model[config.model.encoder.name]

    q_net = q_net_class(**config.model.encoder)

    # decoder
    p_net_class=lookup_model[config.model.decoder.name]
    config.model.decoder.zdim=config.model.zdim
    config.model.decoder.out_channels=config.model.encoder.n_channels
    p_net = p_net_class(**config.model.decoder)

    # vae
    model_class=lookup_model[config.model.name]
    model_kwargs = config.model
    model_kwargs.p_net = p_net
    model_kwargs.q_net = q_net
    model_kwargs.loss_kwargs = config.loss
    model_class=lookup_model[config.model.name]
    model = model_class(**model_kwargs)

    return model

if __name__ == "__main__":
    """ 
    Sample uasge: `python run.py configs.config_o2mnist`
    """ 
    import sys
    import importlib
    import sys
    import wandb
    import run_loops

    # get the config from the command line argument
    if len(sys.argv)>1:
        module = sys.argv[1]
    else:
        raise ValueError("must provide a config, e.g. `python run.py configs.config_o2mnist`")

    print(f"Loading config from {module}")
    config_module=importlib.import_module(module)
    config=config_module.config

    
    # some gloabl sttings
    logging.getLogger().setLevel(logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get datasets specified by config.data
    print(f"Loading data from {config.data.data_dir}")
    dset, loader, dset_test, loader_test = get_datasets_from_config(config)

    # build the model. here specify the number of channels from the dataset
    config.model.encoder.n_channels=dset[0][0].shape[0]  # image channels
    model = build_model_from_config(config)
    print("Model details\n", model.model_details())

    # optimizer - by default, no lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)

    
    if config.do_logging:
        # set up wandb logging, which by default will save stuff locally only. 
        anonymous = "must" if config.wandb_log_settings.wandb_anonymous is None else "allow"
        mode = "offline" if not config.wandb_log_settings.wandb_enbable_cloud_logging else "online"
        wandb.init(config=config, name=config.wandb_log_settings.name, mode=mode, 
                anonymous=anonymous, project=config.wandb_log_settings.project, 
                group=config.wandb_log_settings.group)
        print(f"Logging directory is {wandb.run.dir}")
        print(f"After run is finished, find complete log output in {os.path.join(wandb.run.dir, 'output.log')}")
        fname_save_model = os.path.join(wandb.run.dir, f"model.pt")
        print(f"Find saved models in {fname_save_model}")

    print(f"Running for {config.run.epochs} epochs")
    for epoch in range(config.run.epochs):
        run_loops.train(epoch, model, loader, optimizer, do_progress_bar=config.logging.do_progress_bar, 
               do_wandb=config.do_logging, device=device)

        if config.run.do_validation and epoch%config.run.valid_freq==0:
            run_loops.valid(epoch, model, loader_test, do_progress_bar=config.logging.do_progress_bar,
                    do_wandb=config.do_logging, device=device)
        
        if config.do_logging and epoch%config.logging.checkpoint_epoch==0:
            # by default, overwrite the old model each time
            model.cpu().train()
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),}, fname_save_model)

