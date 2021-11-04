# coding=utf-8
import logging
from os import path, makedirs
import sys

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import itertools

import random
import numpy as np
import math

import pandas as pd

from datasets import load_metric
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

import pdb

TRAINING_ARGS_NAME = "training_args.bin"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"
TOKENIZER_NAME = "tokenizer.json"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_tsa_threshold(t_step, total_step, num_label, mode='linear'):
    """
    Training Signal Annealing (TSA)

    lambda_t (threshold) : TSA threshold
    K (num_label) : # of categries
    t (t_step) : training step
    T (total_step) : total # of training steps

    1. lambda_t = alpha_t * (1 - 1/K) + 1/K

    2. alpha_t by scheduling types
    * linear : t / T
    * log : 1 - exp(-t/T * 5)
    * exp : exp((t/T - 1) * 5)
    """
    if mode == 'linear':
        alpha = t_step / total_step
    elif mode == 'log':
        alpha = 1 - math.exp(-t_step/total_step * 5)
    elif mode == 'exp':
        alpha = math.exp((t_step/total_step - 1) * 5)
    else:
        raise 'No tsa model match!'

    return alpha * (1 - 1/num_label) + 1/num_label


class Trainer:
    def __init__(self, args, config, tokenizer):
        self.args = args
        self.config = config
        self.tokenizer = tokenizer

    def train(self, train_sup_dataset, train_unsup_dataset, eval_dataset, model):
        best_score = -1  # TODO

        model.to(self.args.device)
                
        # multi-gpu training
        if self.args.n_gpu > 1:
            model = nn.DataParallel(model)

        # change to training mode
        model.train()
        
        # calculate batch size for training
        # if you use batch_size 10 with 3 gpus, actual train_batch_size will be 30
        train_batch_size = self.args.per_device_train_batch_size * max(1, self.args.n_gpu)  # consider multi-gpus

        # data loader
        train_sup_sampler = RandomSampler(train_sup_dataset)
        train_sup_dataloader = DataLoader(train_sup_dataset, sampler=train_sup_sampler, batch_size=train_batch_size)
        train_unsup_sampler = RandomSampler(train_unsup_dataset)
        train_unsup_dataloader = DataLoader(train_unsup_dataset, sampler=train_unsup_sampler, batch_size=train_batch_size)
        
        total_step = len(train_unsup_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs if self.args.num_train_steps is None else self.args.num_train_steps
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=total_step)

        
        logger.info("***** Running Training *****")
        logger.info("  Num examples = %d", (len(train_sup_dataset) + len(train_unsup_dataset)))
        logger.info("  Num labeled examples = %d", len(train_sup_dataset))
        logger.info("  Num unlabeled & augmented examples = %d", len(train_unsup_dataset))
        # logger.info("  Num epochs = %d", self.args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_device_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        logger.info("  Gradient accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", total_step)

        train_results = {
            'train_loss': list(),
            'eval_loss': list(),
            'accuracy': list(),
        }
        
        global_step = 0
        train_loss = 0.0
        logging_loss = 0.0
        
        # sup_loss_fn = nn.BCEWithLogitsLoss()   # sup / BCEWithLogitsLoss: sigmoid + BCE
        sup_loss_fn = nn.CrossEntropyLoss(reduction='none')   # sup
        unsup_loss_fn = nn.KLDivLoss(reduction='batchmean')  # unsup

        # fix seed for reproducability
        set_seed(self.args)
        # iter dataloader
        # repreat sup_iter - prevent StopIteration error : iter -> itertools.cycle
        train_sup_iter = itertools.cycle(train_sup_dataloader)
        train_unsup_iter = itertools.cycle(train_unsup_dataloader)

        # for epoch in range(int(self.args.num_train_epochs)):
        for t, unsup_batch in enumerate(tqdm(train_unsup_iter, total=total_step)):
            
            global_step += 1
            
            # load batch
            # supervised data
            sup_batch = next(train_sup_iter)
            sup_batch_tensor = [b.to(self.args.device) for b in sup_batch[0]]
            sup_labels = sup_batch[-1].to(self.args.device).type(torch.long)
            # unsupervised data
            orig_batch_tensor = [b.to(self.args.device) for b in unsup_batch[0]]
            aug_batch_tensor = [b.to(self.args.device) for b in unsup_batch[1]]
            
            # supervised learning
            sup_inputs = {'input_ids': sup_batch_tensor[0], 'attention_mask': sup_batch_tensor[1], 'token_type_ids': sup_batch_tensor[2]}  # 'labels': batch[3] for cross entropy
            sup_outputs = model(**sup_inputs)  # only yield logits
            sup_loss = sup_loss_fn(sup_outputs.logits, sup_labels)

            if self.args.tsa_mode is not None:
                tsa_thres = get_tsa_threshold(global_step, total_step, num_label=self.config.num_labels, mode=self.args.tsa_mode)
                # select probs only exp(log_model_prob) <= tsa_threshold
                tsa_mask = torch.exp(-sup_loss) <= tsa_thres  # why not larger than thres?
                # make mask [1,0,1...,1] which contains 1 for only selected probs 
                sup_loss = sup_loss * tsa_mask
                
            sup_loss = sup_loss.mean()

            # unsupervised learning with data augmentation
            orig_inputs = {'input_ids': orig_batch_tensor[0], 'attention_mask': orig_batch_tensor[1], 'token_type_ids': orig_batch_tensor[2]}
            aug_inputs = {'input_ids': aug_batch_tensor[0], 'attention_mask': aug_batch_tensor[1], 'token_type_ids': aug_batch_tensor[2]}
            
            with torch.no_grad():
                orig_outputs = model(**orig_inputs)

            aug_outputs = model(**aug_inputs)
            aug_probs = aug_outputs.logits / self.args.softmax_temp if self.args.softmax_temp > 0 else 1.
            
            # KL Divergence Loss
            # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html?highlight=kldivloss#torch.nn.KLDivLoss
            # input : log-probabilities 
            # target : probabilities 
            unsup_loss = unsup_loss_fn(F.log_softmax(orig_outputs.logits, dim=-1), F.softmax(aug_probs, dim=-1))
            
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
            
            if global_step % self.args.gradient_accumulation_steps == 0 or global_step >= total_step:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # evaluate during training
            if global_step % self.args.eval_steps == 0 or global_step >= total_step:
                eval_results = self.evaluate(eval_dataset, model)
                
                # logging
                train_results['train_loss'].append(logging_loss/self.args.eval_steps)
                logging_loss = 0.0  # TODO logging step
                train_results['eval_loss'].append(eval_results['eval_loss'])
                train_results['accuracy'].append(eval_results['accuracy'])
                
                model.eval()

            #     if eval_results['accuracy'] > best_score:
            #         best_score = eval_results['accuracy']
            #         # save model (evaluate during training only for dev set)
                self.save(model, optimizer, scheduler, global_step)
                self.save(model, optimizer, scheduler)
                
                model.train()         

            if global_step >= total_step:
                return train_results

    def evaluate(self, eval_dataset, model):
        results = {}
        
        model.to(self.args.device)
        
        # change to evaluation mode
        model.eval()
        
        # calculate batch size for evaluation
        # if you use batch_size 10 with 3 gpus, actual train_batch_size will be 30
        eval_batch_size = self.args.per_device_eval_batch_size * max(1, self.args.n_gpu)  # consider multi-gpus

        # data sampling
        eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
        
        eval_loss = 0.0
        eval_steps = 0

        # fix seed for reproducability
        set_seed(self.args)
        # load metric from datasets (huggingface)
        metric_name = "accuracy"
        metric = load_metric(metric_name)

        for batch in tqdm(eval_dataloader, desc='Evaluation'):

            with torch.no_grad():
                batch_tensor = [b.to(self.args.device) for b in batch[0]]  # batch[0]: orig
                
                inputs = {'input_ids': batch_tensor[0], 'attention_mask': batch_tensor[1], 'token_type_ids': batch_tensor[2], 'labels': batch[-1].to(self.args.device)}
                outputs = model(**inputs)  # only yield logits
                predictions = outputs.logits.argmax(dim=-1).detach().cpu()

                # calculate metrics
                metric.add_batch(predictions=predictions, references=batch[-1])
                
                eval_loss += outputs.loss.mean().item()

            eval_steps += 1

        eval_loss = eval_loss / eval_steps
        results['eval_loss'] = eval_loss
        results.update(metric.compute())

        logger.info(f"  eval_loss : {eval_loss:.2f}")
        logger.info(f"  eval_{metric_name} : {results[metric_name]:.2f}")
        
        return results

    def predict(self, test_dataset):
        pass

    def save(self, model, optimizer, scheduler, global_step=None):
        """
        save model
        ex. output/checkpoint-{global_step}.pt
        """
        if global_step is not None:
            save_dir = path.join(self.args.output_dir, f'checkpoint-{str(global_step)}')
        else:
            save_dir = self.args.output_dir

        try:
            if not path.exists(save_dir):
                makedirs(save_dir)
        except OSError:
            print ('[Error] fail to create directory: ' +  save_dir)

        # from pretrained model
        self.config.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        # from trained model
        torch.save(self.args, path.join(save_dir, TRAINING_ARGS_NAME))
        torch.save(model.state_dict(), path.join(save_dir, WEIGHTS_NAME))
        torch.save(scheduler.state_dict(), path.join(save_dir, SCHEDULER_NAME))
        torch.save(optimizer.state_dict(), path.join(save_dir, OPTIMIZER_NAME))