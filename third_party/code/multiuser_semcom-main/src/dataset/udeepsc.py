import os
import re
import sys
# import mmsdk
import pickle
import torch
import pprint 
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer

from pathlib import Path
# from transformers import *
from tqdm import tqdm_notebook
# from mmsdk import mmdatasdk as md
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK

def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)
    else:
        emb_mat = init_emb
    f = open(path_to_embedding, 'r')
    found = 0
    for line in tqdm_notebook(f, total=embedding_vocab):
        content = line.strip().split()
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        word = ' '.join(content[:-300])
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector
            found += 1
    print(f"Found {found} words in the embedding file.")
    return torch.tensor(emb_mat).float()


class MOSI_dataset():
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))
        
        data_path = str(config.dataset_dir)
        cache_path = data_path + '/embedding_and_mapping.pt'

        try:
            self.train = load_pickle(data_path + '/train.pkl')
            self.dev = load_pickle(data_path + '/dev.pkl')
            self.test = load_pickle(data_path + '/test.pkl')
     
        except:
            print('error')
            pass

    def get_data(self, is_train):
        if is_train:
            return self.train
        else:              
            return self.test


class MSA_dataset(Dataset):
    def __init__(self, config, train=True, shift_offset=0, shuffle_test=False, seed=42):
        dataset = MOSI_dataset(config)
 
        self.data = dataset.get_data(train)
        self.len = len(self.data)

        self.shift_offset = shift_offset  # Shift amount
        self.shuffle_test = shuffle_test  # Whether to shuffle
        self.indices = np.arange(self.len)

        # If shuffling is enabled, apply it
        if shuffle_test:
            np.random.seed(seed)  
            np.random.shuffle(self.indices)  

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        if self.shuffle_test:
            # Use shuffled index if shuffling is enabled
            mapped_index = self.indices[index]
        else:
            # Apply shifting with wrap-around if shuffling is not enabled
            mapped_index = (index - self.shift_offset + self.len) % self.len  

        # Keep text data unchanged
        text_data = self.data[mapped_index][0][0]

        # Fetch modified image and speech data
        image_data = self.data[index][0][1]  
        speech_data = self.data[mapped_index][0][2]  
        
        # Keep targets unchanged
        target = self.data[index][1]  

        return ((text_data, image_data, speech_data, self.data[index][0][3]), target)

    def __len__(self):
        return self.len
    
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class Config_MSA(object):
    def __init__(self, data_dir: Path='data/msadata'):
        project_dir = Path(__file__).resolve().parent.parent.parent
        sdk_dir = project_dir.joinpath('/tmp2/hansliu/data/CMU-MultimodalSDK/')
        data_path = project_dir.joinpath(data_dir)
        data_dict = {'mosi': data_path.joinpath('MOSI'), 'mosei': data_path.joinpath(
            'MOSEI')}
        word_emb_path = '/home/hqyyqh888/SemanRes2/MSA/MISA/glove/glove.840B.300d.txt'
        assert(word_emb_path is not None)
        
        self.dataset_dir = data_dict['mosei']
        self.sdk_dir = sdk_dir
        self.word_emb_path = word_emb_path
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str
    
def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)  
    targets = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    texts = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    images = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], batch_first=True)
    speechs = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], batch_first=True)
    # print(texts.permute(1,0))
    SENT_LEN = texts.size(0)
    # Create bert indices using tokenizer
    bert_details = []
    for sample in batch:
        text = " ".join(sample[0][3])
        encoded_bert_sent = bert_tokenizer.encode_plus(
            text, max_length=SENT_LEN+2, add_special_tokens=True, padding='max_length',truncation=True)
        bert_details.append(encoded_bert_sent)

    bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
    texts = bert_sentences

    return (texts, images, speechs), targets
    
def make_udeepsc_msa_dataloader(batch_size, *args, root='data/msadata', **kwargs):
    """
        Creates dataloaders for the MOSEI dataset tailored for udeepsc learning.
        Args:
            batch_size (int): The number of samples per batch.
            *args: Additional arguments to pass to the dataloader.
            root (str, optional): The root directory where the dataset is stored. Defaults to './data'.
    """

    config_msa = Config_MSA(root)
    dataset = MSA_dataset(config_msa, train=True)

    # define split size
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        dataset    = train_dataset, 
        batch_size = batch_size, 
        *args,
        shuffle    = True,
        drop_last  = True,
        collate_fn = collate_fn,
        **kwargs
    )

    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_dataloader = DataLoader(
        dataset     = val_dataset,
        # sampler     = val_sampler, 
        batch_size  = batch_size, 
        *args,
        shuffle     = False,
        drop_last   = False,
        collate_fn  = collate_fn,
        **kwargs
    )

    train_dataloader.n_iter = len(train_dataloader)
    val_dataloader.n_iter   = len(val_dataloader)

    return train_dataloader, val_dataloader

def make_udeepsc_msa_testdataloader(batch_size, *args, root='data/msadata', **kwargs):
    """
        Creates dataloaders for the MOSEI dataset tailored for udeepsc testing.
        Args:
            batch_size (int): The number of samples per batch.
            *args: Additional arguments to pass to the dataloader.
            root (str, optional): The root directory where the dataset is stored. Defaults to './data'.
    """

    config_msa = Config_MSA(root)
    dataset = MSA_dataset(config_msa, train=False)
    print(f'Total test samples: {len(dataset)}')

    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size, 
        *args,
        pin_memory  = True,
        drop_last   = False,
        collate_fn  = collate_fn, 
        **kwargs
    )

    test_dataloader.n_iter = len(test_dataloader)

    return test_dataloader

if __name__ == "__main__":
    # load dataset
    train_dataloader, val_dataloader = make_udeepsc_msa_dataloader(batch_size=40, root='./data/msadata')
    
    print(len(train_dataloader.dataset))