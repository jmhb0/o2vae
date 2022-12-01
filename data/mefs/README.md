# MEFs dataset
Fluorescence Image of Mouse Embryonic Fibroblast seeded on Circle and Triangle micropatterns. DAPI and Phalloidin. Dataset originally produced for the paper [A robust unsupervised machine-learning method to quantify the morphological heterogeneity of cells and nuclei](https://www.nature.com/articles/s41597-020-00432-x). See that paper for original data source. We downloaded it from [https://github.com/kukionfr/Micropattern_MEF_LMNA_Image](https://github.com/kukionfr/Micropattern_MEF_LMNA_Image).

Using the original images, we did segmentation with manual curation by a cell biologist. Our micrograph segmentations are in [this zenodo link](https://zenodo.org/record/7388245#.Y4k10ezMJqs). The data in this repo is the result of further post-processing: object extraction, and object centering so that each data sample in the data tensor is a single object.

In the home directory, run the folowing to extract the data files:
```
bash data/unzip_mefs.bash
```

Which will have the structure:
```
o2vae/
  data/
    mefs/
      mefs_not_scaled.tar.gz  # compressed preprocessed dataset - scaled 
      mefs_scaled.tar.gz      # compressed preprocessed dataset - not scaled
      unzip_mefs.bash # script to unzip the comppressed data
      X_train.sav     # torch tensor of train images shape (25000,1,128,128)
      y_train.sav     # torch tensor of train labels shape (25000,)
      X_test.sav      # torch tensor of test images shape (6598,1,128,128)
      y_test.sav      # torch tensor of test labels shape (6598,)
```
