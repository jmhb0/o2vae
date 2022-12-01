# o2-mnist
This is a modified version of the classic MNIST dataset, but with random flips and rotations applied. O2-mnist has been used in, for example, [General E(2) - Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251), however the dataset may not be identical - we use our own script to generate it.

In the home directory, build the dataset with:
```
python data/generate_o2mnist.py
```
This script uses `torchvision` package to download the original MNIST and saves it to `o2vae/data/o2_mnist/. It then randomly rotates and flips each image using torchvision transforms. It sets a random seed, so the final dataset is reproducibe.

After running data generation script the structure is:
```
o2vae/
  generate_o2mnist.py
  data/
    o2_mnist/
      X_train.sav  # torch tensor of train images shape (60000,1,32,32))
      y_train.sav  # torch tensor of train labels shape (60000,)
      X_test.sav   # torch tensor of test images shape (10000,1,32,32)
      y_test.sav   # torch tensor of test labels shape (10000,)
      MNIST/       # the original MNIST dataset
```
