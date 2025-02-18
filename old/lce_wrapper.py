from collections import defaultdict
import time
import torch 
import numpy as np
import ir_datasets
import wandb
from torch.autograd import Variable, grad

from transformers import get_linear_schedule_with_warmup, AdamW
from pacednegatives.pairwrapper import PacedWrapper
from pacednegatives.weights import LCEWeights
from pacednegatives.utilities.loss import init_LCEcrossentropy

from torch.utils.data import DataLoader
from accelerate import Accelerator

def batch_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class LCEWrapper():
    def __init__(self, 
                 eta,
                 dataset, 
                 model_name, 
                 batch_size, 
                 model_init, 
                 tokenizer, 
                 lr, 
                 meta_lr,
                 ignore_index,
                 use_mean : bool = None) -> None:

        self.logs = {
            'dataset': dataset,
            'model_name': model_name,
            'batch_size': batch_size,
            'lr': lr,
            'loss' : defaultdict(list),
            'avg_weight' : defaultdict(list),
        }

        self.model = model_init()
        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        self.weights = LCEWeights(eta, device=self.device)
        self.logs['eta'] = []
        self.logs['loss'] = {'main': [], 'meta' : []}
        self.logs['difficulty'] = []
        self.logs['lr'] = {'main': [], 'meta' : []}

        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

        self.loss_fn = init_LCEcrossentropy(ignore_index=ignore_index, use_mean=use_mean)

        self.meta_lr = meta_lr
        self.batch_size = batch_size
        self.meta_optimizer = torch.optim.Adam(self.weights.parameters(), lr=self.meta_lr)

    def create_y(self, x, token='false'):
        y = self.tokenizer([[token]] * len(x), padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids[:, 0].view(-1, 1)
        return Variable(y, requires_grad=False)

    def prep_batch(self, batch):
        px, nx = batch

        px = self.tokenizer(px, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids
        nx = self.tokenizer(nx, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids

        px = Variable(px, requires_grad=False)
        nx = Variable(nx, requires_grad=False)

        op = self.create_y(px, token='true')
        on = self.create_y(nx, token='false')

        return px, nx, op, on


    def meta_loop(self, i):
        px, nx, op, on = self.prep_batch(self.train_loader.get_batch(i, self.difficulty), self.acc_device)
 
        with torch.no_grad():
            plogits = self.model(input_ids=px, labels=op).logits
            nlogits = []
            for _batch in batch_iter(nx, n=int(self.batch_size)):
                nlogits.append(self.model(input_ids=_batch, labels=self.y_neg).logits)
            nlogits = torch.cat(nlogits, dim=0).view(-1, self.train_loader.n, nlogits[0].size(-1)) # Resolve dimensionality issues

        loss = self.loss_fn(plogits, nlogits, op, on, self.weights)
       
        loss.backward()
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()
        self.meta_scheduler.step()

        return loss.item()

    def main_loop(self, i):
        px, nx, op, on = self.prep_batch(self.train_loader.get_batch(i, self.difficulty))
   
        plogits = self.model(input_ids=px, labels=op).logits
        nlogits = []
        for _batch in batch_iter(nx, n=int(self.batch_size)):
            nlogits.append(self.model(input_ids=_batch.to, labels=self.y_neg).logits)
        nlogits = torch.cat(nlogits, dim=0).view(-1, self.train_loader.n, nlogits[0].size(-1)) # Resolve dimensionality issues

        loss = self.loss_fn(plogits, nlogits, op, on)
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return sum([l for l in self.accelerator.gather(loss).tolist() if str(l) != 'nan'])

    def train(self, train_loader, total_steps, warmup_steps):
        torch.manual_seed(42)
        _logger = ir_datasets.log.easy()  
        self.total_steps = total_steps
        self.difficulty = self.weights.eta.item()
        self.train_loader = train_loader
        self.y_neg = self.create_y(torch.ones(self.batch_size,), token='false')
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=warmup_steps // self.train_loader.batch_size if warmup_steps else (total_steps // 100), 
                                                         num_training_steps=total_steps)
        self.meta_scheduler = get_linear_schedule_with_warmup(self.meta_optimizer, 
                                                         num_warmup_steps=warmup_steps // self.train_loader.batch_size if warmup_steps else (total_steps // 100), 
                                                         num_training_steps=total_steps)
        
        start = time.time()
        
        with _logger.pbar_raw(desc=f'train', total=total_steps) as pbar:
            for i in range(total_steps // self.batch_size):
       
                meta_loss = self.meta_loop(i) if self.weights.eta.item() < 1. else 0.
                loss = self.main_loop(i)

                if wandb.run is not None:
                    wandb.log({'loss': loss, 
                               'meta_loss' : meta_loss, 
                               'lr': self.scheduler.get_last_lr()[0], 
                               'meta_lr' : self.meta_scheduler.get_last_lr()[0], 
                               'difficulty': self.difficulty, 
                               'eta' : self.weights.eta.item()})

                self.logs['loss']['main'].append(loss)
                self.logs['loss']['meta'].append(meta_loss)
                self.logs['lr']['main'].append(self.scheduler.get_last_lr()[0])
                self.logs['lr']['meta'].append(self.meta_scheduler.get_last_lr()[0])
                self.logs['difficulty'].append(self.difficulty)
                self.logs['eta'].append(self.weights.eta.item())
              
                self.difficulty = min(self.train_loader.max, self.weights.eta.item())
                pbar.set_postfix({'loss': np.mean(self.logs['loss']['main'])})
                pbar.update(self.train_loader.batch_size)

        end = time.time() - start

        self.logs['time'] = end
        if wandb.run is not None:
            wandb.log({'time': end})

        return self.logs