import torch
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from os import listdir, path, makedirs
from os.path import isfile, join
from enum import Enum
import csv
import pandas as pd
import ast

import logging
import pdb


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class IMDBDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mode: Split = Split.train,
        is_augmented: bool = True,
        is_raw_file: bool = False,
        max_count: int = 100000,
    ):
        """
        data_dir : name of directory
        mode : Split.train/dev/test, use value of this
        is_augmented : load unlabeled data (unsup) if True, else load labeled data (sup)
        max_count : maximum value of examples for test
        TODO add preprocessing
        """
        
        self.orig = None   # original training set
        self.aug = None    # augmented data
        self.label = None  # label for original data
        self.max_count = max_count  # for test

        # print(path.join(txt_file, f'imdb_sup_{mode.value}.txt'))
        # print(path.join(txt_file, f'imdb_unsup_{mode.value}.txt'))
        
        if is_raw_file:
            raise NotImplementedError
        else:
            if is_augmented:
                # use unlabeled data (unsupervised loss)
                data_file_name = f'imdb_unsup_{mode.value}.txt'
                self.orig, _ = self.read_from_file(path.join(data_dir, data_file_name), dtype='ori')
                self.aug, _ = self.read_from_file(path.join(data_dir, data_file_name), dtype='aug')
            else:
                # use labeled dataset (supervised loss)
                data_file_name = f'imdb_sup_{mode.value}.txt'
                self.orig, self.label = self.read_from_file(path.join(data_dir, data_file_name))
    
    def __len__(self):
        return len(self.orig[0])

    def __getitem__(self, index):
        """
        get all tensor data [t1[index], t2[index], .. ] for an item [t1, t2, t3, t4]
        """
        orig  = [tensor[index] for tensor in self.orig] if self.orig is not None else torch.tensor([])
        aug   = [tensor[index] for tensor in self.aug] if self.aug is not None else torch.tensor([])
        label = self.label[index] if self.label is not None else torch.tensor([])
        
        return orig, aug, label

    def read_from_file(self, file_path, dtype=''):
        data = []
        cnt = 0  # test

        logging.info(f'load file at {file_path}')
        header = ['input_ids', 'input_mask', 'input_type_ids']
        header = ['_'.join([dtype, i]) for i in header] if dtype != '' else header

        csv_file = pd.read_csv(file_path, delimiter='\t')
        csv_np = csv_file[header].to_numpy()
        labels = torch.tensor(csv_file['label_ids']) if 'label_ids' in csv_file.keys() else None
        
        for row in csv_np:
            data.append([ast.literal_eval(x) for x in row])
            cnt += 1

            if cnt == self.max_count:
                torch_data = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]
                # pdb.set_trace()
                return torch_data, labels

        torch_data = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]
        
        return torch_data, labels

# --- test --- #
imdb_train_unsup = IMDBDataset("/home/sumin/workspace/uda-impl/data/IMDB/", mode=Split.train, is_augmented=True, max_count=3)
imdb_train_sup = IMDBDataset("/home/sumin/workspace/uda-impl/data/IMDB/", mode=Split.train, is_augmented=False, max_count=3)
imdb_test = IMDBDataset("/home/sumin/workspace/uda-impl/data/IMDB/", mode=Split.test, is_augmented=False, max_count=3)