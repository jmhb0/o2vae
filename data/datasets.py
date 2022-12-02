import torch


class TensorDatasetWithTransform(torch.utils.data.Dataset):
    """
    Same as torch.utils.data.TensorDatset, except a torchvision-style `transform`
    applies to the first tensor.

    Args:
        tensors (tuple): a tuple/array of tensors (any length) that have the
            same length. The first tensor will be data that will be transformed
            by the second argument
        transform: torchvision-style image transform.
    """

    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        """
        return the items from all tensors and apply the transform only to the first
        """
        x = self.tensors[0][index]

        return_object = []
        for i in range(len(self.tensors)):
            x = self.tensors[i][index]
            if i == 0 and self.transform:
                x = self.transform(x)

            return_object.append(x)

        return return_object

    def __len__(self):
        return self.tensors[0].size(0)
