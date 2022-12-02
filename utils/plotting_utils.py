import torch
import numpy as np
import matplotlib.pyplot as plt
import glasbey
import seaborn as sns

def plot_embedding_space_w_labels(X, y, figsize=(9,9),
            scatter_kwargs=dict(s=0.1,legend_fontsize=10, legend_marker_size=100, hide_labels=True),
            colormap="glasbey",
):
    """
    Make a 2d scatter plot of an embedding space (e.g. umap) colored by labels.
    """
    assert X.shape[1]==2
    f,axs = plt.subplots(figsize=(9,9))

    if colormap=="glasbey":
        colors = glasbey.create_palette(palette_size=10)
    else:
        colors = sns.color_palette("tab10")
    y_uniq = np.unique(y)
    for i, label in enumerate(y_uniq):
        idxs = np.where(y==label)[0]
        axs.scatter(X[idxs,0], X[idxs,1],
                    color=colors[i], s=scatter_kwargs['s'], label=i)

    legend=plt.legend(fontsize=scatter_kwargs['legend_fontsize'])
    [legend.legendHandles[i].set_sizes([scatter_kwargs['legend_marker_size']], dpi=300)
             for i in range(len(legend.legendHandles))]

    if scatter_kwargs['hide_labels']:
        axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.close()

    return f, axs

def get_embedding_space_embedded_images(embedding, data, n_yimgs=70, n_ximgs=70,
            xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Given an embedding (e.g. umap or tsne embedding) and their original images, 
    generate an image (that can be passed to plt.imshow) that samples images from
    the space. This is similar to the tensorflow embedding projector.

    How it works: break space into grid. Each image is assigned to a rectangles if it's
    enclosed by that rectangle. The image nearest the rectangle centroid (L2
    distance) is assigned to it, and plotted.

    Args
      embedding: Array shape (n_imgs, 2,) holding the tsne embedding coordinates
        of the imgs stored in data
      data: original data set of images. len(data)==len(embedding). To be plotted
        on the TSNE grid.
    Returns
        img_plot (Tensor): the tensor to pass to `plt.imshow()`. 
        object_indices: the indices of the imgs in `data` corresponding to the 
            grid points.
    """
    assert len(data)==len(embedding)
    img_shape = data.shape[-2:]
    ylen, xlen = data.shape[-2:]
    if xmin is None: xmin=embedding[:,0].min()
    if xmax is None: xmax=embedding[:,0].max()
    if ymin is None: ymin=embedding[:,1].min()
    if ymax is None: ymax=embedding[:,1].max()

    # Define grid corners
    ycorners, ysep = np.linspace(ymin, ymax, n_yimgs, endpoint=False, retstep=True)
    xcorners, xsep = np.linspace(xmin, xmax, n_ximgs, endpoint=False, retstep=True)
    # centroids of the grid
    ycentroids=ycorners+ysep/2
    xcentroids=xcorners+xsep/2

    # determine which point in the grid each embedded point belongs
    img_grid_indxs = (embedding - np.array([xmin, ymin])) // np.array([xsep,ysep])
    img_grid_indxs = img_grid_indxs.astype(dtype=int)

    #  Array that will hold each points distance to the centroid
    img_dist_to_centroids = np.zeros(len(embedding))

    # array to hold the final set of images
    img_plot=torch.zeros(n_yimgs*img_shape[0], n_ximgs*img_shape[1])

    # array that will give us the returnedindices
    object_indices=torch.zeros((n_ximgs, n_yimgs), dtype=torch.int)

    # Iterate over the grid
    for i in range(n_ximgs):
        for j in range(n_yimgs):
            ## Get indices of points that are in this box
            indxs=indxs = np.where(
                    np.all(img_grid_indxs==np.array([i,j])
                    ,axis=1)
                )[0]

            ## calculate distance to centroid for each point
            centroid=np.array([xcentroids[i],ycentroids[j]])
            img_dist_to_centroids[indxs] = np.linalg.norm(embedding[indxs] - centroid, ord=2, axis=1)

            ## Find the nearest image to the centroid
            # if there are no imgs in this box, then skip
            if len(img_dist_to_centroids[indxs])==0:
                indx_nearest=-1
            # else find nearest
            else:
                # argmin over the distances to centroid (is over a restricted subset)
                indx_subset = np.argmin(img_dist_to_centroids[indxs])
                indx_nearest = indxs[indx_subset]
                # Put image in the right spot in the larger image
                xslc = slice(i*xlen, i*xlen+xlen)
                yslc = slice(j*ylen, j*ylen+ylen)
                img_plot[xslc, yslc] = torch.Tensor(data[int(indx_nearest)])

            # save the index
            object_indices[i,j] = indx_nearest

    # turns out the x and y coordiates got mixed up so I have to transpose it here
    # and also I need to flip the image
    img_plot = torch.transpose(img_plot, 1,0)
    img_plot = torch.flip(img_plot,dims=[0])

    return img_plot, object_indices
