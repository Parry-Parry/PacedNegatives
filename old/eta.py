from collections import defaultdict
import logging
import time
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import ir_datasets

from torch.autograd import Variable, grad
from transformers import AdamW, get_linear_schedule_with_warmup
from pacednegatives.weights import EtaWeights, Weights
from pacednegatives.pairwrapper import PacedWrapper

class MetaWrapper(PacedWrapper):
    def __init__(self, 
                 eta,
                 min_eta,
                 max_eta,
                 dataset, 
                 model_name, 
                 batch_size, 
                 model_init, 
                 tokenizer, 
                 lr, 
                 ignore_index) -> None:
        super().__init__(dataset, model_name, batch_size, model_init, tokenizer, lr, ignore_index)

        self.weights = EtaWeights(eta, device=self.device, min=np.log(min_eta), max=max_eta)
        self.logs['eta'] = eta

    def meta_loop(self, j):
        px, nx, o_p, o_n = self.prep_batch(self.train_loader.get_batch(j, self.weights[j]))

        self.meta_model.load_state_dict(self.model.state_dict())

        ## positive
        
        logits = self.meta_model(input_ids=px, labels=o_p).logits
        ce = self.loss_fn(logits.view(-1, logits.size(-1)), o_p.view(-1))
        v = self.weights.forward(loss=ce, idx=j)

        weighted_ce_p = torch.sum(ce * v) / len(ce)

        ## negative

        logits = self.meta_model(input_ids=nx, labels=o_n).logits
        ce = self.loss_fn(logits.view(-1, logits.size(-1)), o_n.view(-1))
        self.weights.eta = self.weights.clamp(self.weights.eta)
        v = self.weights.forward(loss=ce, idx=j)
        weighted_ce_n = torch.sum(ce * v) / len(ce)
        
        weighted_ce = weighted_ce_p + weighted_ce_n
        
        self.meta_model.zero_grad()
        grads = grad(weighted_ce, (self.meta_model.parameters()), create_graph=True, retain_graph=True)
        self.update_params(self.meta_model, lr=self.scheduler.get_lr(), grads=grads)
        del grads

        ## update weights

        with torch.no_grad():
            logits = self.meta_model(input_ids=px, labels=o_p).logits
        ce = self.loss_fn(logits.view(-1, logits.size(-1)), o_p.view(-1))
        v = self.weights.forward(loss=ce, idx=j)
        weighted_ce_p = torch.sum(ce * v) / len(ce)

        with torch.no_grad():
            logits = self.meta_model(input_ids=nx, labels=o_n).logits
        ce = self.loss_fn(logits.view(-1, logits.size(-1)), o_n.view(-1))
        v = self.weights.forward(loss=ce, idx=j)
        weighted_ce_n = torch.sum(ce * v) / len(ce)

        weighted_ce = weighted_ce_p + weighted_ce_n - torch.sum(v)
        grads = grad(weighted_ce, (v,), create_graph=True, retain_graph=True)

        v_ce = v - self.scheduler.get_lr() * grads[0]
        self.weights.set_weight(idx=j, weight=v_ce)
        del grads

    def main_loop(self, j):
        px, nx, o_p, o_n = self.prep_batch(self.train_loader.get_batch(j, self.weights[j]))

        self.weights.eta = self.weights.clamp(self.weights.eta)

        logits = self.model(input_ids=px, labels=o_p).logits
        ce = self.loss_fn(logits.view(-1, logits.size(-1)), o_p.view(-1))
        with torch.no_grad():
            v = self.weights.forward(loss=ce, idx=j)
        weighted_ce_p = torch.sum(ce * v) / len(ce)

        logits = self.meta_model(input_ids=nx, labels=o_n).logits
        ce = self.loss_fn(logits.view(-1, logits.size(-1)), o_n.view(-1))
        with torch.no_grad():
            v = self.weights.forward(loss=ce, idx=j)
        weighted_ce_n = torch.sum(ce * v) / len(ce)

        weighted_ce = weighted_ce_p + weighted_ce_n
        
        loss = torch.mean(weighted_ce)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train(self, train_loader, epochs, warmup_steps):
        self.epochs = epochs
        self.train_loader = train_loader

        torch.manual_seed(RND)
        _logger = ir_datasets.log.easy()

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=warmup_steps if warmup_steps else len(train_loader) // 4, 
                                                         num_training_steps=epochs * len(train_loader))

        start = time.time()

        for epoch in range(epochs):
            with _logger.pbar_raw(desc=f'train {epoch}', total= len(train_loader) // self.batch_size) as pbar:
                for j in range(len(train_loader) // self.batch_size):
                    self.meta_loop(j)
                    eta, ce, weighted_ce, v = self.main_loop(j)

                    if j == 1:  logging.info(f'loss: {ce} | v : {v}')

                    total_loss += weighted_ce.item()

                    if j % 100 == 0: logging.info(f'BATCH: {j} | Average v: {torch.mean(v).item()} | eta: {eta.item()}')

                    self.logs['eta'][epoch].append(eta.item())
                    self.logs['loss'][epoch].append(weighted_ce.item())
                    self.logs['avg_weight'][epoch].append(torch.mean(v).item())
                
                    pbar.update(1)
                    pbar.set_postfix({'loss': total_loss / (j+1)})

        end = time.time() - start

        self.logs['time'] = end

        return self.logs
