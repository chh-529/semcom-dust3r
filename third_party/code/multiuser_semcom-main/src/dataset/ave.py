import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

class AVEDataset(Dataset):
    """
        From https://github.com/YapengTian/AVE-ECCV18/blob/master/dataloader.py: AVEDataset
        Translate to pytorch dataset by me
    """
    def __init__(self, data_dir, split):

        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        # self.add_infra = add_infra
        self.data_path = str(data_dir)
        self.image_dir = self.data_path + '/visual_feature.h5'
        self.audio_dir = self.data_path + '/audio_feature.h5'

        self.label_dir = self.data_path + '/labels.h5'
        self.order_dir = self.data_path + f'/{split}_order.h5'

        # Load order
        with h5py.File(self.order_dir, 'r') as hf:
            self.order = hf['order'][:]
        
        # Load features and labels
        with h5py.File(self.audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:]
        with h5py.File(self.label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:]
        with h5py.File(self.image_dir, 'r') as hf:
            self.video_features = hf['avadataset'][:]

    def __len__(self):
        return len(self.order)
    
    def __getitem__(self, index):
        vid_idx = self.order[index]
        
        video = torch.tensor(self.video_features[vid_idx], dtype=torch.float32) # (time_steps, 7, 7, 512)
        audio = torch.tensor(self.audio_features[vid_idx], dtype=torch.float32) # (time_steps, 128)
        label = torch.tensor(self.labels[vid_idx], dtype=torch.float32)
         # (time_steps, num_class)
        return (video, audio), label
        
    def get_batch(self, idx):
        """ Deprecated """
        for i in range(self.batch_size):
            id = idx * self.batch_size + i

            self.video_batch[i, :, :, :, :] = self.video_features[self.lis[id], :, :, :, :]
            self.audio_batch[i, :, :] = self.audio_features[self.lis[id], :, :]
            self.label_batch[i, :, :] = self.labels[self.lis[id], :, :]

        return torch.from_numpy(self.video_batch).float(), torch.from_numpy(self.audio_batch).float(), torch.from_numpy(self.label_batch).float()

def make_AVE_dataloaders(batch_size, *args, root='data/avedata', do_torch_normalize=True, **kwargs):
    """
        Creates dataloaders for the MF dataset tailored for chaneel fusion learning.
        Args:
            batch_size (int): The number of samples per batch.
            *args: Additional arguments to pass to the dataloader.
            root (str, optional): The root directory where the dataset is stored. Defaults to './data'.
            **kwargs: Additional keyword arguments to pass to the dataloader.
        
        Returns:
            tuple: A tuple containing the training and testing dataloaders.
                - train_dataloader: DataLoader for the training dataset.
                - val_dataloader: DataLoader for the validating dataset.

    """
    train_dataset = AVEDataset(root, split='train')
    val_dataset = AVEDataset(root, split='val')

    train_dataloader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        *args,
        drop_last   = True, 
        **kwargs
    )

    val_dataloader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        *args,
        drop_last   = False, 
        **kwargs
    )

    return train_dataloader, val_dataloader

def make_AVE_testdataloader(batch_size, *args, root='data/avedata', do_torch_normalize=True, **kwargs):
    """
        Creates dataloaders for the MOSEI dataset tailored for udeepsc testing.
        Args:
            batch_size (int): The number of samples per batch.
            *args: Additional arguments to pass to the dataloader.
            root (str, optional): The root directory where the dataset is stored. Defaults to './data'.
    """
    test_dataset = AVEDataset(root, split='test')
    print(f'Total test samples: {len(test_dataset)}')

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        *args,
        drop_last   = False, 
        **kwargs
    )

    test_dataloader.n_iter = len(test_dataloader)

    return test_dataloader


if __name__ == '__main__':
    data_dir = 'data/avedata'
    
    # load dataset
    train_dataloader, _ = make_AVE_dataloaders(batch_size=30, root=data_dir)
    
    from ..utils import str_type
    for it, batch in enumerate(train_dataloader):
        print(str_type(batch))
        if it >= 5:
            break