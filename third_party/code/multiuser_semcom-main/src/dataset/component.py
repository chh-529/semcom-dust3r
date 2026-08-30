import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from typing import *
from pathlib import Path
from ..utils import str_type

def make_multiuser_dataloader(dataset, batch_size, n_user, *args, collate_function: Callable = None, **kwargs):
    """
        Make a dataloader that returns a batch of data for each user.

        this will make dataloader have a batch size of `batch_size * n_user`,
        but we will make it return (batch_size, n_user, channel, h, w) instead.
        i.e., give each user `batch_size` amount of data

        will set this keyword for dataloader:
            collate_fn=collate_fn (in function),
            drop_last=True (every batch must be batch_size * user)

        Args:
            dataset: yields (tensor with shape (channel, h, w), label: int) (or something that is NOT a dict)
            batch_size: the batch size for each user
            n_user: number of user
            *args, **kwargs: arguments for dataloader
    """
    if collate_function is None:
        collate_function = torch.utils.data.default_collate

    def collate_fn(batch):
        """
            batch: a list of data from dataset, expect (tensor with shape (channel, h, w), label: int)
        """
        # tensors = torch.utils.data.default_collate(batch)
        tensors = collate_function(batch)
        return [
            tensor.view(batch_size, n_user, *tensor.size()[1:])
            for tensor in tensors
        ]

    return torch.utils.data.DataLoader(
        dataset, batch_size * n_user, *args, collate_fn=collate_fn, drop_last=True, **kwargs
    )

def make_multiuser_multidataset_dataloader(datasets, batch_size, *args, **kwargs):
    """
        Make a dataloader that returns a batch of data for each user.
        Each batch comes from different datasets, as specified in `datasets`

        this will make dataloader have a batch size of `batch_size * n_user`,
        but we will make it return (batch_size, n_user, channel, h, w) instead.
        i.e., give each user `batch_size` amount of data

        will set this keyword for dataloader:
            collate_fn=collate_fn (in function),
            drop_last=True (every batch must be batch_size * user)

        Args:
            datasets: a list of dataset,
                      each yields (tensor with shape (channel, h, w), label: int) (or something that is NOT a dict).
                      make sure each dataset yields the same (channel, h, w) shape!
                      assume n_user = len(datasets)
            batch_size: the batch size for each user
            *args, **kwargs: arguments for dataloader
    """
    pass

    # class MultiuserDataset(Dataset):
    #     def __init__(self, datasets, batch_size):
    #         self.datasets = datasets

    #     def __len__(self):
    #         return min([len(dataset) for dataset in self.datasets])

    #     def __getitem__(self, idx):
    #         return [dataset[idx] for dataset in self.datasets]
    
    # def collate_fn(batch):
    #     """
    #         batch: a list of data from dataset, expect (tensor with shape (channel, h, w), label: int)
    #     """
    #     tensors = torch.utils.data.default_collate(batch)
    #     return [
    #         tensor.view(batch_size, n_user, *tensor.size()[1:])
    #         for tensor in tensors
    #     ]

    # return torch.utils.data.DataLoader(
    #     dataset, batch_size * n_user, *args, collate_fn=collate_fn, drop_last=True, **kwargs
    # )

class RandomSampleDataset(Dataset):
    """
        random samples the given dataset
        the index doesn't have any meaning
    """
    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        random_idx = torch.randint(0, len(self.dataset), (1,)).item()
        return self.dataset[random_idx]
    

if __name__ == '__main__':
    pass