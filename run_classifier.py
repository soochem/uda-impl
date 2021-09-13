# coding=utf-8
import logging
import os
import sys
import pdb

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import random
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from enum import Enum

from utils.data_loader import IMDBDataset
from src.trainer import Trainer

import pdb

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class ModelArguments:
    """
    Additional arguments for fine-tuning models
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataArguments:
    """
    Additional arguments for fine-tuning models to which data
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


if __name__ == "__main__":
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load data
    # train -> unsup (6 cols) / sup (4 cols)
    train_dataset = (
        IMDBDataset(
            data_dir=data_args.data_dir,
            # tokenizer=tokenizer,
            # max_seq_length=data_args.max_seq_length,
            mode=Split.train,
            has_label=True,
        )
        if training_args.do_train
        else None
    )

    # split train:dev = 8:2
    train_size = int(0.8 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, eval_size], 
        generator=torch.Generator().manual_seed(training_args.seed)
    )

    test_dataset = (
        IMDBDataset(
            data_dir=data_args.data_dir,
            # tokenizer=tokenizer,
            # max_seq_length=data_args.max_seq_length,
            mode=Split.test,  # temp
            has_label=True,
        )
        if training_args.do_train
        else None
    )

    # train
    num_labels = 1  # binary classfication -> MSE loss

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    trainer = Trainer(training_args, tokenizer, model)


    # train
    if training_args.do_train:
        trainer.train(train_dataset, eval_dataset)

    # evaluate
    if training_args.do_eval:
        trainer.evaluate(eval_dataset)