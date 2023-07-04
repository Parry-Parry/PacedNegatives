from collections import defaultdict
import logging
import time
from typing import List
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import ir_datasets

from torch.autograd import Variable, grad
from transformers import AdamW, get_linear_schedule_with_warmup
from pacednegatives.weights import Weights

class ParaWrapper:
    def __init__(self, 
                 dataset,
                 model_name,
                 batch_size,
                 model_init,
                 tokenizer,
                 lr, 
                 ignore_index,
                 gpus : List[int] = [0, 1]) -> None:
        cudnn.benchmark = True

        self.logs = {
            'dataset': dataset,
            'model_name': model_name,
            'batch_size': batch_size,
            'lr': lr,
            'loss' : defaultdict(list),
            'avg_weight' : defaultdict(list),
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

        self.model = nn.DataParallel(model_init(), gpu_ids=gpus).to(0)
        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        self.batch_size = batch_size

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def update_params(self, model, lr, grads):
        for tgt, src in zip(model.named_parameters(), grads):
            name_t, param_t = tgt
            grad = src
            tmp = nn.Parameter(param_t - lr * grad, requires_grad=True)
            self.set_param(model, name_t, tmp)

    def prep_batch(self, batch):
        px, nx, o_p, o_n = batch

        px = self.tokenizer(px, padding=True, return_tensors='pt').input_ids.to(0)
        nx = self.tokenizer(nx, padding=True, return_tensors='pt').input_ids.to(0)
        o_p = self.tokenizer(o_p, padding=True, return_tensors='pt').input_ids[:, 0].view(-1, 1).to(0)
        o_n = self.tokenizer(o_n, padding=True, return_tensors='pt').input_ids[:, 0].view(-1, 1).to(0)

        px = Variable(px, requires_grad=False)
        nx = Variable(nx, requires_grad=False)
        o_p = Variable(o_p, requires_grad=False)
        o_n = Variable(o_n, requires_grad=False)

        return px, nx, o_p, o_n

class ParaStdWrapper(ParaWrapper):
    def __init__(self, 
                 dataset, 
                 model_name, 
                 batch_size, 
                 model_init, 
                 tokenizer, 
                 lr, 
                 ignore_index,
                 gpus) -> None:
        super().__init__(dataset, model_name, batch_size, model_init, tokenizer, lr, ignore_index, gpus)

    def meta_loop(self, j):
        px, nx, o_p, o_n = self.prep_batch(self.train_loader.get_batch(j, self.weights[j]))

        with torch.no_grad():
            logits = self.model(input_ids=px, labels=o_p).logits
        ce = self.loss_fn(logits.view(-1, logits.size(-1)), o_p.view(-1))
        v = self.weights.forward(idx=j)
        weighted_ce_p = torch.sum(ce * v) / len(ce) 

        with torch.no_grad():
            logits = self.model(input_ids=nx, labels=o_n).logits
        ce = self.loss_fn(logits.view(-1, logits.size(-1)), o_n.view(-1))
        v = self.weights.forward(idx=j)
        weighted_ce_n = torch.sum(ce * v) / len(ce)

        weighted_ce = weighted_ce_p + weighted_ce_n - torch.sum(v)
        grads = grad(weighted_ce, (v,), create_graph=True, retain_graph=True)

        v_ce = v - self.scheduler.get_last_lr()[0] * grads[0]
        self.weights.set_weight(idx=j, val=v_ce)
        del grads

    def main_loop(self, j):
        px, nx, o_p, o_n = self.prep_batch(self.train_loader.get_batch(j, self.weights[j]))

        logits = self.model(input_ids=px, labels=o_p).logits
        ce_p = self.loss_fn(logits.view(-1, logits.size(-1)), o_p.view(-1))

        logits = self.model(input_ids=nx, labels=o_n).logits
        ce_n = self.loss_fn(logits.view(-1, logits.size(-1)), o_n.view(-1))

        loss = torch.mean(ce_p) + torch.mean(ce_n)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train(self, train_loader, epochs, warmup_steps):
        torch.manual_seed(42)
        _logger = ir_datasets.log.easy()
        self.weights = Weights((len(train_loader) // self.batch_size, self.batch_size), device=self.device)
        self.train_loader = train_loader   
        self.epochs = epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=warmup_steps if warmup_steps else len(train_loader) // 4, 
                                                         num_training_steps=epochs * len(train_loader))
        
        start = time.time()

        for epoch in range(epochs):
            with _logger.pbar_raw(desc=f'train {epoch}', total=len(train_loader) // self.batch_size) as pbar:
                for j in range(len(train_loader) // self.batch_size):
                    self.meta_loop(j)
                    loss = self.main_loop(j)

                    self.logs['loss']['train'].append(loss)
                    self.logs['avg_weight'][epoch].append(self.weights[j].mean().item())  

                    pbar.update(1)

        end = time.time() - start

        self.logs['time'] = end