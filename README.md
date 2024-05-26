 ![Black](https://img.shields.io/badge/code%20style-black-000000.svg) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-red.svg)](https://www.python.org/downloads/release/python-360/) ![GitHub](https://img.shields.io/github/license/jmhb0/o2vae) [![MARVL](https://img.shields.io/badge/Stanford-MARVL-820000)](https://marvl.stanford.edu/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7388245.svg)](https://doi.org/10.5281/zenodo.7388245) 

# O2VAE - orientation invariant morphologic profiling for cells and organelles

This repo contains source code and demos for our paper ["Orientation-invariant autoencoders learn robust representations for shape profiling of cells and organelles"](https://www.nature.com/articles/s41467-024-45362-4). 

## Contents
- [Background and Method](#method)
- [What's in this repo](#contents)
- [Usage - learning representations](#usage1)
- [Usage - using representation for analysis](#usage2)
- [Citation](#citation)

## <a name="method"/> Background and method
In phenotypic profiling for cell biology, we want to map images of centered cells or organelles to a vector of numbers (a profile / representation / embedding / feature). We then use those vectors for analyses like clustering, classification, outlier detection, dimensionality reduction, and visualization:

![image - paper Fig.1b](./assets/representation_learning_tasks.svg)

If we naively apply unsupervised learning methods like principal component analysis (PCA) or autoencoders, then rotating the image changes the representation vector (below left). Trying to do clustering with these vectors may give bad results. Instead, we want the output vector to be the same for any rotation or flip of the image (below, right). This is called O(2)-invariance.

<p align="center">
<img src="./assets/representation-sensitivity_v2_86ec4e50_crop.gif" width="640" height="112" class="center">
</p>

Our representation learning method, O2-VAE, enforces O2-invariance. It is a deep autoencoder that is trained to compress the image to a vector and then reconstruct it. After training, the compressed vector is used as the morphologic profile. 

![image - o2vae Fig.1a](./assets/o2vae-model.png)


## <a name="contents"> What's in this repo
**Learning representations: O2-VAE model and training methods**
Code for defining and training the O2-VAE model based on PyTorch (see [usage - learning](#usage1)).  Orientation invariance is enforced by the model architecure, using the [e2cnn](https://github.com/QUVA-Lab/e2cnn/) library. 

**Using representations: analysis and visualization tools for cell biology profiling**
Notebooks demonstrating example analyses (see [this section](#usage2)). Extracting learned representations from a pretrained model and examples of clustering, outlier detection, classficiation, dimensionality reduction, and visualization. 

**Efficient image registration: module for rotation and flip image alginment on GPUs**  
The O2-VAE loss function requires finding the rotation and flip that best aligns two images, and [Reddy et al](https://ieeexplore.ieee.org/abstract/document/506761) propose an efficient Fourier-based method. We provide an implementation that takes advantage of efficient batch processing on GPUs, which may be useful for other computer vision applications (see `./registration/` and its [guide](./registration/README.md))

**Prealignment methods** 
For very simple datasets (e.g. nuclei segmentation masks) a preprocessing method, 'prealignment', may be enough to control for orientation sensitivty. For users who want to try this approach before using O2vae, We provide some basic functions (see `./prealignment/` and its [guide](./prealignment/README.md)).


## <a name="usage1"/> Usage - learning representations 
### Installation 
We tested the following on linux ubuntu 20.04.5 LTS with Python3.9. Recommend creating a conda environment: 
```
conda create --name o2vae python=3.9
conda activate o2vae
```
Install standard packages (time <5mins):
```
pip install -r requirements.txt
```
Next go to to [pytorch](https://pytorch.org/) (section "INSTALL PYTORCH") to install the correct torch, torchvision, and cuda versions. As a reference, we save our environment for testing in `environment.yml`.

The model training is much faster with access to GPUs, which can be accessed freely using [Colab](https://research.google.com/colaboratory/faq.html). 

### Configuration
`./configs/` has example config files. See the  file's comments for more about changing default data locations, model architecture, loss functions, and logging parameters.

### Datasets 
The scripts will search a directory (defined in config file `config.data.data_dir`) for datasets. It must have at least `X_train.sav`, which should be a numpy array or torch Tensor containing images of centered objects. The array shape is `(n_samples,n_channels,height,width)`. Optionally, you can have test data, `X_test.sav` for validation during training. You can also provide labels `y_train.sav` and `y_test.sav`.


### Logging and saving models
We use [weights and biases](https://wandb.ai/) to handle logging. Each run will create a new folder inside `wandb/<run_name>` containing the saved model in `wandb/<run_name>/files/model.pt` (printed to screen after running). 

[optional] To access the wandb dashboard with training metrics, log in to a weights and biases account and set the config file to:  
```
config.wandb_log_settings.wandb_anonymous=False 
config.wandb_log_settings.wandb_enbable_cloud_logging=True
```

### Scripts for model training 
To train an o2-vae model, edit `./run.bash` to point to the right config file, and run:
```
bash run.bash
```
The example commands in that script are for the demo dataset and configs (mext section). Training these demos on GPUs (nvidia-rtx) with the default configs in `run.bash` takes <1min per training epoch for both demo datasets. Training converges in about 50 epochs.
 
**Important** check the terminal for the location of the saved models. Something like:
> Logging directory is `wandb/<log_dir>`

### Demos
We provide two demo datasets, [o2-mnist](./data/o2_mnist/README.md) and [MEFS](./data/mefs/README.md). To get these datasets run:
```
python data/generate_o2mnist.py
bash data/mefs/unzip_mefs.bash
```
They each have a config file `configs/config_o2mnst.py` and `configs/config_mefs.py`. A model can be trained using the script above, OR they can be run in notebooks `examples/`

### Running in a notebook
Examples notebooks for training models are in `examples/`. This is mostly the same code as `run.py` but without any logging. 

## <a name="usage2"/> Usage - using representation for analysis  
### Recovering trained models 
Take the same model config file, `configs/<my_config>`, and get saved model location, `fname_model=wandb/<run_name>/files_model.pt`. Then you can recover the model with:

```
import run
import torch

from configs.<my_config> import config
model=run.get_datasets_from_config(config)

fname_model=wandb/<run_name>/files_model.pt
saved_model=torch.load(fname_model)
model.load_state_dict(saved_model['state_dict'])
```

### Extracting learned features / representations
Load the dataset from the config information and extract features:

```
import run 
from utils import utils
from configs.<my_config> import config

dset, loader, dset_test, loader_test = run.get_datasets_from_config(config)

embeddings, labels = utils.get_model_embeddings_from_loader(model, loader, return_labels=True)
embeddings_test, labels_test = utils.get_model_embeddings_from_loader(model, loader_test, return_labels=True)
```
Note that downstream analysis only needs the representations; you do not need access to the model. 

### Anlaysis 
See `examples/` for notebooks with example analysis, which use functions in `utils/`.

## <a name="citation"/> Citation
If this repo contributed to your research, please consider citing our paper:
```
@ARTICLE{Burgess2024-zb,
  title     = "Orientation-invariant autoencoders learn robust representations
               for shape profiling of cells and organelles",
  author    = "Burgess, James and Nirschl, Jeffrey J and Zanellati, Maria-Clara
               and Lozano, Alejandro and Cohen, Sarah and Yeung-Levy, Serena",
  journal   = "Nat. Commun.",
  publisher = "Springer Science and Business Media LLC",
  volume    =  15,
  number    =  1,
  pages     = "1022",
  month     =  feb,
  year      =  2024,
}
```

