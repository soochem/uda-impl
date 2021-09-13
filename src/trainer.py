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
    # AutoConfig,
    # AutoModelForTokenClassification,
    # AutoModel,
    # AutoTokenizer,
    # EvalPrediction,
    # HfArgumentParser,
    # Trainer,
    # TrainingArguments,
    AdamW,
    get_linear_schedule_with_warmup,
)

from enum import Enum

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


class Trainer:
    def __init__(self, args, tokenizer, model):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model

    def train(self, train_dataset, eval_dataset):
        best_score = -1
        train_losses = []
        
        # change to training mode
        self.model.to(self.args.device)
        self.model.train()
        
        # calculate batch size for training
        # if you use batch_size 10 with 3 gpus, actual train_batch_size will be 30
        train_batch_size = self.args.per_device_train_batch_size * max(1, self.args.n_gpu)  # consider multi-gpus

        # data sampling
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        
        global_step_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=global_step_total)
        
        # multi-gpu training
        if self.args.n_gpu > 1:
            model = nn.DataParallel(self.model)
        
        logger.info("***** Running Training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num epochs = %d", self.args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_device_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        logger.info("  Gradient accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", global_step_total)

        global_step = 0
        train_loss = 0.0
        logging_loss = 0.0
        stop_training = False
        
        loss_fn = nn.CrossEntropyLoss()

        # fix seed for reproducability
        set_seed(self.args)

        for epoch in range(int(self.args.num_train_epochs)):
            
            for batch in train_dataloader:
                global_step += 1
                batch = [b.to(self.args.device) for b in batch]
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}  # 'labels': batch[3] for cross entropy
                outputs = model(**inputs)  # only yield logits
                loss = loss_fn(outputs[0], batch[-1].unsqueeze(-1))
                
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()            
                train_loss += loss.item()
                # pdb.set_trace()

                if global_step % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()   
                    model.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                if global_step % self.args.eval_steps == 0:
                    model.eval()

                    cur_f1 = self.evaluate(self.args, model if self.args.n_gpu==1 else model.module, eval_dataset)
                    
                    model.train()

            if stop_training:
                break

    def evaluate(self, eval_dataset):
        best_score = -1
        eval_losses = []
        
        # change to training mode
        self.model.to(self.args.device)
        self.model.eval()
        
        # calculate batch size for training
        # if you use batch_size 10 with 3 gpus, actual train_batch_size will be 30
        eval_batch_size = self.args.per_device_train_batch_size * max(1, self.args.n_gpu)  # consider multi-gpus

        # data sampling
        eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
        
        global_step_total = len(eval_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
        # multi-gpu training
        if self.args.n_gpu > 1:
            model = nn.DataParallel(self.model)
        
        logger.info("***** Running Evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_device_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            eval_dataset
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        logger.info("  Gradient accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", global_step_total)

        global_step = 0
        train_loss = 0.0
        eval_loss = 0.0
        logging_loss = 0.0

        # fix seed for reproducability
        set_seed(self.args)

        for epoch in range(int(self.args.num_train_epochs)):
            
            for batch in eval_dataloader:
                global_step += 1
                batch = [b.to(self.args.device) for b in batch]
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}  # 'labels': batch[3] for cross entropy
                outputs = model(**inputs)  # only yield logits

                

    def predict(self, test_dataset):
        pass
