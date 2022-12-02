import numpy as np
import matplotlib.pyplot as plt
import torch

import sklearn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from torchvision.utils import make_grid

def do_clusterering(embeddings, n_clusters, random_state=0, do_cluster_centers=1, n_pca=32):
    """
    Do GMM and Kmeans clustering. Also get the cluster centroids and a 'score' that is
    the confidence a sample should be in its cluster. For GMM this is the probability.
    For Kmeans this is the negative squared distance from the centroid.
    """
    ### GMM clustering. Do it in pca-reduced space for cimputational saving
    #      (but you should check that `cls_gmm.explained_variance_` is high for your dataset.
    pca = PCA(n_components=n_pca, svd_solver='arpack', random_state=random_state).fit(embeddings)
    embeddings_pca_reduced = pca.fit_transform(embeddings)
    cls_gmm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(embeddings_pca_reduced)
    labels_gmm = cls_gmm.predict(embeddings_pca_reduced)

    ## kmeans
    cls_kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(embeddings)
    labels_kmeans = cls_kmeans.labels_

    ## compute  cluster centers
    centers_gmm = pca.inverse_transform(cls_gmm.means_) ## map back to the original space
    centers_kmeans = cls_kmeans.cluster_centers_

    ## compute 'scores' of each data point. For GMM it's proabbility. For kmeans it's squared distance from cluster centroid
    ## where we just take the negative of the distance from the centroid
    scores_gmm = cls_gmm.score_samples(embeddings_pca_reduced)
    kmeans_center_per_label = centers_kmeans[labels_kmeans]
    scores_kmeans = -np.linalg.norm(kmeans_center_per_label-embeddings.numpy(), ord=2, axis=1)**2

    return (labels_gmm, labels_kmeans), ((pca, cls_gmm), cls_kmeans),\
            (centers_gmm, centers_kmeans), (scores_gmm, scores_kmeans)


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Source: https://github.com/sgvaze/generalized-category-discovery 

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def make_sample_grid_for_clustering(labels_img, data_imgs, scores, method=None, n_examples=10,
        stds_filt=1, verbose=True, paper_figure_grid=False):
    """
    Make the final clustering group grid by getting the indexes
    calling `do_sampling`
    `scores` are some metric for assessing the likelihood some image belongs to a cluster,
    and these methods rae kind of shaky.
    """
    k = len(np.unique(labels_img))
    img_shape=data_imgs.shape[1:]
    sample_imgs = np.zeros((k,n_examples,*img_shape))
    uniq_labels, cnts = np.unique(labels_img, return_counts=1)

    for i, l in enumerate(uniq_labels):
        # idx_samples=np.argwhere(labels_img==l)[:n_examples,0]
        idx_samples = do_sampling(l, labels_img, scores, method=method, n_examples=n_examples, stds_filt=stds_filt, verbose=verbose)
        n_examples_actual = len(idx_samples) # in case there isn't enough

        imgs= data_imgs[idx_samples]
        sample_imgs[i, :n_examples_actual] = imgs

    # flatten along everything but the image dims
    sample_imgs = np.reshape(sample_imgs, (-1,*img_shape))
    grid = make_grid(torch.Tensor(sample_imgs), n_examples, pad_value=0.5).moveaxis(0,2)

    return grid, cnts

def do_sampling(l, labels_img, scores, method=None, n_examples=10, stds_filt=1, verbose=True):
    """
    Various sampling strategies for the clustering, indicated by "method".
    Called by `make_sample_grid_for_clustering`
    Methods
        None: just take the first n_examples in the list.
    """
    idx_samples=np.argwhere(labels_img==l)[:,0]

    if method is None:
        idx_samples = idx_samples[:n_examples]
        pass

    # return the highest-scoring things
    elif method=="top":
        # confusing, but it works (note the trailing underscores)
        scores_ = scores[idx_samples]
        idx_samples_ = np.flip(np.argsort(scores_))
        idx_samples = idx_samples[idx_samples_]

    # return the highest-scoring things
    elif method=="bottom":
        # confusing, but it works (note the trailing underscores)
        scores_ = scores[idx_samples]
        idx_samples_ = np.argsort(scores_)
        idx_samples = idx_samples[idx_samples_]

    # get elements within `stds_filt` of the mean of scores in this cluster
    elif method=="std":
        idx_samples=np.argwhere(labels_img==l)[:,0]
        scores_ = scores[idx_samples]
        mean, std = np.mean(scores_), np.std(scores_)
        n_before = len(scores_)
        scores_in_range = (scores_>=(mean-std*stds_filt)) & (scores_<=(mean+std*stds_filt))
        idx_samples=idx_samples[scores_in_range]
        n_after  = len(idx_samples)
        if verbose:
            print(f"STD dev reduction removes {100*(1-n_after/n_before):.0f}% of points")

    elif method in ("uniform", "uniform_partial"):
        # first order them, as in "top"
        scores_ = scores[idx_samples]
        idx_samples_ = np.flip(np.argsort(scores_))
        idx_samples = idx_samples[idx_samples_]

        # now sample uniformly from the ordered list of sample ids
        if method=="uniform":
            idx_uniform = np.linspace(0,len(idx_samples), n_examples).astype(int)
        # unfiform but stop early up to the 80th perdentile
        elif method=="uniform_partial":
            idx_uniform = np.linspace(0,int(len(idx_samples)*.8), n_examples).astype(int)
        idx_uniform[-1]=idx_uniform[-1]-1
        idx_samples = idx_samples[idx_uniform]

    return idx_samples[:n_examples]

def purity_score(y_true, y_pred):
    """ https://stackoverflow.com/questions/34047540/python-clustering-purity-metric """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

