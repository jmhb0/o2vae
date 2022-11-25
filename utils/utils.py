import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def get_model_embeddings_from_loader(model, loader, return_labels=False, 
        do_progress_bar=True, device='cuda'):
    """
    """
    model.eval().to(device)
    embeddings = []
    labels = [] if return_labels else None
    with torch.no_grad():
        iterator = tqdm.tqdm(loader, position=0, leave=True) if do_progress_bar else loader
        for batch in iterator:
            z = model.embed_data(batch[0].cuda()).cpu()
            embeddings.append(z)
            if return_labels:
                labels.append(batch[1])
    embeddings = torch.cat(embeddings)
    if return_labels: labels = torch.cat(labels)

    return embeddings, labels

def get_model_embeddings_from_tensors(model, tensor, tensor_labels=None, batch_size=64, 
        do_progress_bar=True, device='cuda', return_labels=False):
    """
    Get the embeddings and (optionally) labels from a large tensor (and labels). 
    It is run batchwise.
    """
    # if labels provided, make the dataset with it, else don't
    if tensor_labels is not None: 
        assert not return_labels
        dset = TensorDataset(tensor, tensor_labels)
    else:
        dset = TensorDataset(tensor)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
    return get_model_embeddings_from_loader(model, loader, return_labels=return_labels,
            do_progress_bar=do_progress_bar, device='cuda')

class Bunch(dict):
    """ 
    From https://stackoverflow.com/questions/38034377/object-like-attribute-access-for-nested-dictionary
    Dictionary subclass whose entries can be accessed by attributes (as well
        as normally).
    Useful for config objects.
    >>> obj = AttrDict()
    >>> obj['test'] = 'hi'
    >>> print obj.test
    hi
    >>> del obj.test
    >>> obj.test = 'bye'
    >>> print obj['test']
    bye
    >>> print len(obj)
    1
    >>> obj.clear()
    >>> print len(obj)
    0
    """
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_nested_dicts(cls, data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})

def plot_sample_data(loader, nrow=10, ncol=10, figsize=(9,9), pad_value=0.5):
    x, y = next(iter(loader))
    grid = make_grid(x[:nrow*ncol], nrow, pad_value=pad_value).moveaxis(0,2)
    f,axs = plt.subplots(figsize=figsize)
    axs.imshow(grid)
    axs.set_axis_off()
    plt.close()
    return f, axs 
