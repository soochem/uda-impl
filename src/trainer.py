# coding=utf-8
import logging
import os
import sys
import pdb

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import random
import numpy as np
# from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import load_metric
import torch
from torch import nn
import torch.nn.functional as F
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

import pdb

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class Trainer:
    def __init__(self, args):
        self.args = args

    def train(self, train_sup_dataset, train_unsup_dataset, eval_dataset, model):
        best_score = -1  # TODO
        # train_losses = []
        
        model.to(self.args.device)
                
        # multi-gpu training
        if self.args.n_gpu > 1:
            model = nn.DataParallel(model)

        # change to training mode
        model.train()
        
        # calculate batch size for training
        # if you use batch_size 10 with 3 gpus, actual train_batch_size will be 30
        train_batch_size = self.args.per_device_train_batch_size * max(1, self.args.n_gpu)  # consider multi-gpus

        # data sampling
        train_sup_sampler = RandomSampler(train_sup_dataset)
        train_sup_dataloader = DataLoader(train_sup_dataset, sampler=train_sup_sampler, batch_size=train_batch_size)
        train_unsup_sampler = RandomSampler(train_unsup_dataset)
        train_unsup_dataloader = DataLoader(train_unsup_dataset, sampler=train_unsup_sampler, batch_size=train_batch_size)
        
        global_step_total = len(train_unsup_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=global_step_total)

        
        logger.info("***** Running Training *****")
        logger.info("  Num examples = %d", (len(train_sup_dataloader) + len(train_unsup_dataloader)))
        logger.info("  Num labeled examples = %d", len(train_sup_dataloader))
        logger.info("  Num unlabeled & augmented examples = %d", len(train_unsup_dataloader))
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
        
        sup_loss_fn = nn.BCEWithLogitsLoss()   # sup / BCEWithLogitsLoss: sigmoid + BCE
        unsup_loss_fn = nn.KLDivLoss(reduction='batchmean')  # unsup

        # fix seed for reproducability
        set_seed(self.args)

        for epoch in range(int(self.args.num_train_epochs)):
            train_sup_iter = iter(train_sup_dataloader)

            for unsup_batch in tqdm(train_unsup_dataloader):
                global_step += 1

                # load batch
                # supervised data
                sup_batch = next(train_sup_iter)
                sup_batch_tensor = [b.to(self.args.device) for b in sup_batch[0]]
                sup_labels = sup_batch[-1].to(self.args.device).type(torch.float)
                # unsupervised data
                orig_batch_tensor = [b.to(self.args.device) for b in unsup_batch[0]]
                aug_batch_tensor = [b.to(self.args.device) for b in unsup_batch[1]]
                
                # supervised learning
                sup_inputs = {'input_ids': sup_batch_tensor[0], 'attention_mask': sup_batch_tensor[1], 'token_type_ids': sup_batch_tensor[2]}  # 'labels': batch[3] for cross entropy
                sup_outputs = model(**sup_inputs)  # only yield logits
                sup_loss = sup_loss_fn(sup_outputs.logits, sup_labels.unsqueeze(-1))

                # unsupervised learning with data augmentation
                orig_inputs = {'input_ids': orig_batch_tensor[0], 'attention_mask': orig_batch_tensor[1], 'token_type_ids': orig_batch_tensor[2]}
                aug_inputs = {'input_ids': aug_batch_tensor[0], 'attention_mask': aug_batch_tensor[1], 'token_type_ids': aug_batch_tensor[2]}
                
                with torch.no_grad():
                    orig_outputs = model(**orig_inputs)
                aug_outputs = model(**aug_inputs)
                
                # KL Divergence Loss
                # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html?highlight=kldivloss#torch.nn.KLDivLoss
                # input : log-probabilities 
                # target : probabilities 
                unsup_loss = unsup_loss_fn(F.log_softmax(orig_outputs.logits, dim=-1), F.softmax(aug_outputs.logits, dim=-1))
                
                # combine supervised + unsupervised loss
                loss = sup_loss + self.args.loss_weight * unsup_loss

                # aggregate multi-gpu result
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                # backprop
                loss.backward()            
                train_loss += loss.item()
                logging_loss += loss.item()
                
                if global_step % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()   
                    model.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                
                # evaluate during training
                if global_step % self.args.eval_steps == 0:
                    model.eval()

                    cur_acc = self.evaluate(eval_dataset, model)
                    
                    if cur_acc['accuracy'] > best_score:
                        best_score = cur_acc
                    
                    model.train()

    def evaluate(self, eval_dataset, model):
        eval_losses = []
        results = {}
        
        # change to evaluation mode
        model.eval()
        
        # calculate batch size for evaluation
        # if you use batch_size 10 with 3 gpus, actual train_batch_size will be 30
        eval_batch_size = self.args.per_device_train_batch_size * max(1, self.args.n_gpu)  # consider multi-gpus

        # data sampling
        eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
        
        logger.info("***** Running Evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_device_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            eval_batch_size
        )

        global_step = 0
        eval_loss = 0.0

        # fix seed for reproducability
        set_seed(self.args)
        # load metric from datasets (huggingface)
        metric = load_metric("accuracy")

        for batch in eval_dataloader:
            global_step += 1
            batch_tensor = [b.to(self.args.device) for b in batch[0]]  # batch[0]: orig
            inputs = {'input_ids': batch_tensor[0], 'attention_mask': batch_tensor[1], 'token_type_ids': batch_tensor[2]}  # 'labels': batch[3] for cross entropy
            outputs = model(**inputs)  # only yield logits
            predictions = outputs.logits.argmax(dim=-1)

            # calculate metrics
            metric.add_batch(predictions=predictions, references=batch[-1])

        results.update(metric.compute())

        return results


    def predict(self, test_dataset):
        pass
