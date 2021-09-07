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
    AdamW,
    get_linear_schedule_with_warmup,
)

from enum import Enum
from data_loader import IMDBDataset

import pdb

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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


class Trainer:
    def __init__(self):
        pass

    def train(self, args, train_dataset, eval_dataset, model):
        global_step = 0
        best_score = -1
        train_losses = []
        
        # change to training mode
        model.to(args.device)
        model.train()
        
        # calculate batch size for training
        # if you use batch_size 10 with 3 gpus, actual train_batch_size will be 30
        train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)  # consider multi-gpus

        # data sampling
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        
        global_step_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=global_step_total)
        
        # multi-gpu training
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
        
        logger.info("***** Running training *****")
        logger.info("  Num Examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", global_step_total)

        global_step = 0
        train_loss = 0.0
        logging_loss = 0.0
        stop_training = False
        
        loss_fn = nn.CrossEntropyLoss()

        # fix seed for reproducability
        set_seed(args)

        for epoch in range(int(args.num_train_epochs)):
            
            for batch in train_dataloader:
                global_step += 1
                batch = [b.to(args.device) for b in batch]
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}  # 'labels': batch[3] for cross entropy
                outputs = model(**inputs)  # only yield logits
                loss = loss_fn(outputs[0], batch[-1].unsqueeze(-1))
                
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                loss.backward()            
                train_loss += loss.item()
                # pdb.set_trace()

                if global_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()   
                    model.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                if global_step % args.eval_steps == 0:
                    model.eval()

                    cur_f1 = self.evaluate(args, model if args.n_gpu==1 else model.module, eval_dataset)
                    
                    model.train()

            if stop_training:
                break

    def evaluate(self, args, eval_dataset, model):
        pass

    def predict(self, args, test_dataset, model):
        pass


# --- test --- #
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# load data
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

eval_dataset = (
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
    # id2label=label_map,
    # label2id={label: i for i, label in enumerate(labels)},
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
)
model = AutoModelForTokenClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
)

Trainer().train(training_args, train_dataset, eval_dataset, model)