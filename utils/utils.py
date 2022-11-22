import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm


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
