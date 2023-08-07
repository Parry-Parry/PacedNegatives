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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model_init()
        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        self.accelerator = Accelerator()

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

    def create_y(self, x, token='false', cpu=False):
        y = self.tokenizer([token] * len(x), padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids[:, 0].view(-1, 1)
        if not cpu: y = y.to(self.device)
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


    def meta_loop(self, px, nx):
        px, nx, op, on = self.prep_batch((px, nx))
 
        with torch.no_grad():
            plogits = self.model(input_ids=px, labels=op).logits
            nlogits = []
            for _batch in batch_iter(nx, n=int(self.batch_size)):
                nlogits.append(self.model(input_ids=_batch.to(self.device), labels=self.y_neg).logits)
            nlogits = torch.cat(nlogits, dim=0).view(-1, self.train_loader.n, nlogits[0].size(-1)) # Resolve dimensionality issues

        loss = self.loss_fn(plogits, nlogits, op, on, self.weights)
       
        loss.backward()
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()
        self.meta_scheduler.step()

        return loss.item()

    def main_loop(self, px, nx):
        px, nx, op, on = self.prep_batch((px, nx))
   
        plogits = self.model(input_ids=px.to(self.device), labels=op).logits
        nlogits = []
        for _batch in batch_iter(nx, n=int(self.batch_size)):
            nlogits.append(self.model(input_ids=_batch.to(self.device), labels=self.y_neg).logits)
        nlogits = torch.cat(nlogits, dim=0).view(-1, self.train_loader.n, nlogits[0].size(-1)) # Resolve dimensionality issues

        loss = self.loss_fn(plogits, nlogits, op, on)
        
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return sum([l for l in self.accelerator.gather(loss).tolist() if str(l) != 'nan'])

    def train(self, train_loader, total_steps, warmup_steps):
        torch.manual_seed(42)
        _logger = ir_datasets.log.easy()  
        self.total_steps = total_steps
        self.difficulty = self.weights.eta.item()
        self.y_neg = self.create_y(torch.ones(self.batch_size,))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=warmup_steps // self.train_loader.batch_size if warmup_steps else (total_steps // 100), 
                                                         num_training_steps=total_steps)
        self.meta_scheduler = get_linear_schedule_with_warmup(self.meta_optimizer, 
                                                         num_warmup_steps=warmup_steps // self.train_loader.batch_size if warmup_steps else (total_steps // 100), 
                                                         num_training_steps=total_steps)
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, train_loader, self.scheduler, device_placement=[True, True, False, False])
        self.train_loader.dataset.weight = self.difficulty
        start = time.time()
        
        with _logger.pbar_raw(desc=f'train', total=total_steps) as pbar:
            for px, nx in self.train_loader:
                self.train_loader.dataset.weight = self.difficulty
                meta_loss = self.meta_loop(px, nx) if self.weights.eta.item() < 1. else 0.
                self.train_loader.dataset.weight = self.difficulty
                loss = self.main_loop(px, nx)

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
              
                self.difficulty = min(self.train_loader.dataset.max, self.weights.eta.item())
                pbar.set_postfix({'loss': np.mean(self.logs['loss']['main'])})
                pbar.update(self.train_loader.batch_size)

        end = time.time() - start

        self.logs['time'] = end
        if wandb.run is not None:
            wandb.log({'time': end})

        return self.logs