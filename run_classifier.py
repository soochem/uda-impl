# coding=utf-8
import logging
import os
import sys

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from argparse import ArgumentParser

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
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
class DataTrainingArguments(TrainingArguments):
    """
    Additional arguments for fine-tuning models to which data and from which model.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weighting factor to balance supervised cross entropy loss and unsupervised consistency training loss"}
    )


if __name__ == "__main__":
    
    parser = HfArgumentParser(DataTrainingArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # if args are written as json file, load args from json
        training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        training_args = parser.parse_args_into_dataclasses()

    training_args = training_args[0]

    num_labels = 1  # binary classfication -> MSE loss

    config = AutoConfig.from_pretrained(
        training_args.model_name_or_path,
        num_labels=num_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        training_args.model_name_or_path,
        from_tf=bool(".ckpt" in training_args.model_name_or_path),
        config=config,
    )

    
    # load data
    # train -> unsup (6 cols) / sup (4 cols)
    # train -> sup 20 + unsup 60k
    MAX_COUNT = 20  # for test
    
    train_sup_dataset = (
        IMDBDataset(
            data_dir=training_args.data_dir,
            # tokenizer=tokenizer,
            # max_seq_length=data_args.max_seq_length,
            mode=Split.train,
            is_augmented=False,
            max_count=MAX_COUNT,
        )
        if training_args.do_train
        else None
    )
    
    train_unsup_dataset = (
        IMDBDataset(
            data_dir=training_args.data_dir,
            # tokenizer=tokenizer,
            # max_seq_length=data_args.max_seq_length,
            mode=Split.train,
            is_augmented=True,
            max_count=MAX_COUNT,
        )
        if training_args.do_train
        else None
    )

    # # split train:dev = 8:2
    # train_size = int(0.8 * len(train_unsup_dataset))
    # eval_size = len(train_unsup_dataset) - train_size
    
    # TODO eval dataset
    # train_dataset, eval_dataset = torch.utils.data.random_split(
    #     train_unsup_dataset, 
    #     [train_size, eval_size], 
    #     generator=torch.Generator().manual_seed(training_args.seed)
    # )

    test_dataset = (
        IMDBDataset(
            data_dir=training_args.data_dir,
            # tokenizer=tokenizer,
            # max_seq_length=data_args.max_seq_length,
            mode=Split.test,  # temp
            is_augmented=False,
            max_count=MAX_COUNT,
        )
        if training_args.do_eval
        else None
    )
    
    trainer = Trainer(training_args)
    
    # train
    if training_args.do_train:
        trainer.train(train_sup_dataset, train_unsup_dataset, test_dataset, model)
    
    # evaluate
    if training_args.do_eval:
        trainer.evaluate(test_dataset, model)