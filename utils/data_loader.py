import torch
from torch.utils.data import Dataset, DataLoader

from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from os import listdir, path, makedirs
from os.path import isfile, join
from enum import Enum
import csv
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
        has_label: bool = True,
        is_raw_file: bool = False,
        max_count: int = 10000,
    ):
        """
        data_dir : name of directory
        mode : Split.train/dev/test, use value of this
        has_label : load labeled data (sup) if True, else load unlabeled data (unsup)
        max_count : maximum value of examples for test
        TODO add preprocessing
        """
        
        data = []
        self.tensor_data = None

        # print(path.join(txt_file, f'imdb_sup_{mode.value}.txt'))
        # print(path.join(txt_file, f'imdb_unsup_{mode.value}.txt'))
        
        if is_raw_file:
            raise NotImplementedError
        else:
            if has_label:
                # use labeled dataset (supervised loss)
                data_file_name = f'imdb_sup_{mode.value}.txt'
            else:
                # use unlabeled data (supervised loss)
                data_file_name = f'imdb_unsup_{mode.value}.txt'
            
            with open(path.join(data_dir, data_file_name), 'r', encoding='utf-8') as f:
                imdb_reader = csv.reader(f, delimiter='\t', quotechar='"')  # read txt as csv format splited by tab and excluding character "

                cnt = 0  # test
                header = next(imdb_reader, None)
                # print(header)
                
                for row in imdb_reader:
                    # row : list of lists
                    data.append([ast.literal_eval(x) for x in row])
                    cnt += 1

                    if cnt == max_count:
                        self.tensor_data = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]  # zip data in the same row, across col
                        print(self.tensor_data)
                        return

                self.tensor_data = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]  # zip data in the same row, across col
                return
    
    def __len__(self):
        return len(self.tensor_data[0])

    def __getitem__(self, index):
        """
        get all tensor data [t1[index], t2[index], .. ] for an item [t1, t2, t3, t4]
        """
        return [tensor[index] for tensor in self.tensor_data]


# --- test --- #
IMDBDataset("/home/sumin/workspace/uda-impl/data/IMDB/", mode=Split.train, has_label=True, max_count=3)
IMDBDataset("/home/sumin/workspace/uda-impl/data/IMDB/", mode=Split.train, has_label=False, max_count=3)
# IMDBDataset("/home/sumin/workspace/uda-impl/data/IMDB/", Split.test, has_label=True)