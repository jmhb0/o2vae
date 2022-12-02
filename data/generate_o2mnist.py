# usage:
#    `python generate_o2mnist.py`
# In <home>/data/o2_mnist, will created 2 folders - first `MNIST` (handled by torchvision).
# and a second, `o2_mnist` which is build from MNIST in the code below. It will have the same
# data except rotated and flipped unifornmly at random. '

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as T_f


"""
try:
    data_dir = sys.argv[1]
    assert os.path.exists(data_dir)
except Exception as e:
    print("pass an existing data directory argument: `python generate_o2mnist.py /path/to/data_dir`")
    raise ValueError(e)
    """
data_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(data_dir, "o2_mnist")
print(f"O2-mnist data dir: {data_dir}")

torch.manual_seed(0)
seed = np.random.randint(1)
random.seed(seed)

Path(data_dir).mkdir(exist_ok=True)

import torchvision.transforms as T
import tqdm
from torch.utils.data import DataLoader


train = True

# we use torchvision transforms for the random rotations and flips.
# we also resize to a larger image before flipping, which should reduce
# interpolation artifacts.
add_padding = 2  # We want 32-dim images by doing padding=2
transform = T.Compose(
    (
        T.ToTensor(),
        T.Pad(add_padding),
        T.Resize(32 * 3),
        T.RandomRotation(360),
        T.RandomVerticalFlip(0.5),
        T.Resize(32),
    )
)


def get_o2_mnist(train=True):
    dset = torchvision.datasets.MNIST(
        root=data_dir, download=True, transform=transform, train=train
    )
    # batch size needs to be 1 so that we resample the transform rotation and flip for each image
    loader = DataLoader(dset, batch_size=1, shuffle=False)
    X, y = [], []

    for batch in tqdm.tqdm(loader):
        X.append(batch[0])
        y.append(batch[1])

    X, y = torch.cat(X), torch.cat(y)
    return X, y


# generate the O2 dataset
print("Generating O2-vae train set")
X_train, y_train = get_o2_mnist(train=True)
print("Generating O2-vae test set")
X_test, y_test = get_o2_mnist(train=False)

torch.save(X_train, os.path.join(data_dir, "X_train.sav"))
torch.save(y_train, os.path.join(data_dir, "y_train.sav"))
torch.save(X_test, os.path.join(data_dir, "X_test.sav"))
torch.save(y_test, os.path.join(data_dir, "y_test.sav"))
