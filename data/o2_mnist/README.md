# o2-mnist
This is a modified version of the classic MNIST dataset, but with random flips and rotations applied. (O2-mnist has been used in, for example, [General E(2) - Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251)).

In the home directory, build the dataset with:
```
python data/generate_o2mnist.py
```
This script uses `torchvision` package to download the original MNIST and saves it to `o2vae/data/o2_mnist/. It then randomly rotates and flips each image using torchvision transforms. It sets a random seed, so the final dataset is reproducibe.

The training output data and labels are saved to `X_train.sav`, `y_train.sav`. The test set is `X_test.sav` and `y_test.sav`. The train /test split is the same as the original dataset. 
