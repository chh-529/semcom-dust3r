# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from ..util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0), 
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]

# from ipdb import set_trace as st

class MF_dataset(Dataset):
    """
        From https://github.com/haqishen/MFNet-pytorch/blob/master/util/MF_dataset.py
    """

    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640 ,transform=[]):
        super(MF_dataset, self).__init__()

        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.is_train  = have_label
        self.n_data    = len(self.names)
        self.n_class   = 9


    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image     = np.array(Image.open(file_path)) # (w,h,c)
        image.flags.writeable = True
        return image

    def get_train_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')

        for func in self.transform:
            image, label = func(image, label)

        image = np.array(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        label = Image.fromarray(label.astype(np.uint8)).resize((self.input_w, self.input_h), resample=Image.NEAREST)
        label = np.array(label, dtype=np.int64)

        image, label = torch.tensor(image), torch.tensor(label)
        # label[label > 8] = 0
        # label = torch.clamp(label, min=0, max=(self.n_class - 1))
        
        return image, label

    def get_test_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        image = np.array(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255

        return torch.tensor(image), name


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else: 
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data
    
def make_channelfusion_MF_dataloaders(batch_size, *args, root='./data/MF/', do_torch_normalize=True, **kwargs):
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
        Yields:
            tuple: A tuple containing:
                - image tensor with shape (batch_size, channel, h, w)
                - label tensor with shape (batch_size, num_class)
    """
    train_dataset = MF_dataset(root, 'train', have_label=True, transform=augmentation_methods)
    val_dataset  = MF_dataset(root, 'val', have_label=True)

    train_dataloader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        *args,
        pin_memory  = True,
        drop_last   = True, 
        **kwargs
    )
    val_dataloader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        *args,
        pin_memory  = True,
        drop_last   = False, 
        **kwargs
    )

    train_dataloader.n_iter = len(train_dataloader)
    val_dataloader.n_iter   = len(val_dataloader)

    return train_dataloader, val_dataloader


def make_channelfusion_MF_testdataloader(batch_size, *args, root = './data/MF', **kwargs):
    """
        Creates test dataloader for the MF dataset tailored for chaneel fusion learning.
        Args:
            batch_size (int): The number of samples per batch.
            *args: Additional arguments to pass to the dataloader.
            root (str, optional): The root directory where the dataset is stored. Defaults to './data'.
            **kwargs: Additional keyword arguments to pass to the dataloader.
        Returns:
            test_dataloader: DataLoader for the testing dataset.
        Yields:
            tuple: A tuple containing:
                - image tensor with shape (batch_size, channel, h, w)
                - label tensor with shape (batch_size, num_class)
    """
    test_dataset  = MF_dataset(root, 'test', have_label=True)
    print(f'Total test samples: {len(test_dataset)}')
    
    test_dataloader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        *args,
        pin_memory  = True,
        drop_last   = False, 
        **kwargs
    )
    test_dataloader.n_iter = len(test_dataloader)

    return test_dataloader


from ..utils import str_type
from ..util.util import visualize
from pathlib import Path
from tqdm import tqdm
if __name__ == '__main__':
    result_folder = Path('./tmp/label_test')
    result_folder.mkdir(parents=True, exist_ok=True)
    
    data_dir = './data/MF/'
    name = '00812N'
    # MF = MF_dataset(data_dir, 'train', have_label=True, transform=augmentation_methods)
    dataloader, _ = make_channelfusion_MF_dataloaders(batch_size=100)

    def read_image(name, folder):
        file_path = os.path.join(data_dir, '%s/%s.png' % (folder, name))
        image     = np.array(Image.open(file_path)) # (w,h,c)
        # image.flags.writeable = True

        return image
    
    image = read_image(name, 'images')
    label = read_image(name, 'labels')

    for func in augmentation_methods:
        image, label = func(image, label)

    print (f'Val: {set(torch.tensor(label.copy()).flatten().tolist())}')

    label = Image.fromarray(label.astype(np.uint8)).resize((480, 640), resample=Image.NEAREST)
    label = np.array(label, dtype=np.int64)
    label = torch.tensor(label)
    unique_values, counts = torch.unique(label, return_counts=True)
    print(f'{unique_values= }')
    print(f'{counts= }')

    # progress_bar = tqdm(enumerate(dataloader), leave=False, total=len(dataloader), dynamic_ncols=True)
    # outsiders = 0
    # for batch_i, (_, labels, names) in progress_bar:
    #     filt = labels.amax(dim=(1,2)) > 8
    #     labels = labels[filt]
    #     out_names = [n for n, m in zip(names, filt.tolist()) if m]
    #     fpaths = [(result_folder/(n + '_label.png')) for n in out_names]
        
    #     outsiders += len(labels)
    #     visualize(fpaths, labels)
        
    # print(f"Total out labels: {outsiders}") # For training data: 194/1500, validation: 0