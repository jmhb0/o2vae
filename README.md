# O2VAE - orientation invariant morphologic profiling for cells and organelles

This repo contains code and demos for our paper [Learning orientation-invariant representations enables accurate and robust morphologic profiling of cells and organelles](https://biorxiv.org/). 

## Contents
[Background and Method](#method)
[What's in this repo](#contents)
[Usage - learning representations](#usage1)
[Usage - using representation for analysis](#usage2)
[Citation](#citation)

## <a name="method"/> Background and method
In morphologic profiling for cell biology, we want to map image of centered cells or organelles to a vector of numbers (a profile/representation/embedding). We then use those vectors for analyses like clustering, classification, outlier detection, dimensionality reduction, and visualization. 
![image - paper Fig.1b](./assets/representation_learning_tasks.png)

If we naively apply unsupervised learning methods like PCA or autoencoders, then rotating the image changes the representation vector. 
![image vae orientation-sensitivity](./assets/orientation_sensitive_representation.gif)

Our method representation learning method, O2-VAE, forces the embeddings to be the same under any rotation or flip (this is the group O(2) of orthogonal transforms):
![image o2vae invariance](./assets/o2vae-representation.gif)

Orientation invariance improves downstream anaylses. For example, we cluster representation spaces learned with (left) and without (right) enforcing orientation invariance. Clusters from O2-invariant representations are based on shape, but clusters from non-invariant representations are sometimes based on orientation as well. 
![image - good and bad clustering](./assets/mefs_clustering_samples.png)

The O2-VAE is a deep autoencoder that is trained to compress the image to a vector and then to reconstruct it. After training, the compressed vector is used as the morphologic profile. 
![image - o2vae Fig.1a](./assets/o2vae-model.png)


## <a name="contents"> What's in this repo
**Learning representations: O2-VAE model and training methods**
Code for defining and training the O2-VAE model based on PyTorch (see [usage - learning](#usage1).  Orientation invariance is enforced by the model arhcitecure, using the [e2cnn](https://github.com/QUVA-Lab/e2cnn/) library. 

**Using representations: analysis and visualization tools for cell biology profiling**
Notebooks demonstrating example analyses (see [this section](#usage2)). Extracting learned representations from a pretrained model and examples of clustering, outlier detection, classficiation, dimensionality reduction, and visualization. 

**Efficient image registration: module for rotation and flip image alginment on GPUs**  
The O2-VAE loss function requires finding the rotation and flip that best aligns two images, and [Reddy et al](https://ieeexplore.ieee.org/abstract/document/506761) propose an efficient Fourier-based method. We provide an implementation that takes advantage of efficient batch processing on GPUs, which may be useful for other computer vision applications (see `registration/` and its [guide](./registration/README.md)

**Prealignment methods** 
For very simple datasets (e.g. nuclei segmentation masks) a preprocessing method, 'prealignment', may be enough to control for orientation sensitivty. For users who want to try this approach before using O2vae, We provide some basic functions (see `prealignment/` and its [guide](./prealignment/README.md).

##<a name="usage1"/> Usage - learning representations 

## <a name="usage2"/> Usage - using representation for analysis  

## <a name="citation"/> Citation
f this repo contributed to your research, please consider citing our paper:
```
```

