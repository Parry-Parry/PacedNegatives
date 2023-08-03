import time
import torch 
import numpy as np
import ir_datasets
import wandb
from torch.autograd import Variable, grad

from transformers import get_linear_schedule_with_warmup
from pacednegatives.pairwrapper import PacedWrapper
from pacednegatives.weights import EtaWeights
from pacednegatives.utilities.loss import LCEcrossentropy

class LCEWrapper(PacedWrapper):
    def __init__(self, 
                 eta,
                 dataset, 
                 model_name, 
                 batch_size, 
                 model_init, 
                 tokenizer, 
                 lr, 
                 meta_lr,
                 ignore_index) -> None:
        super().__init__(dataset, model_name, batch_size, model_init, tokenizer, lr, ignore_index)

        self.weights = EtaWeights(eta, device=self.device, min=0.+1e-10, max=1.)
        self.logs['eta'] = []
        self.logs['loss'] = {'main': [], 'meta' : []}
        self.logs['difficulty'] = []
        self.logs['lr'] = {'main': [], 'meta' : []}
        self.logs['success_rate'] = []
        self.logs['probs'] = []

        self.running_rate = []

        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

        self.loss_fn = LCEcrossentropy(ignore_index=ignore_index)

        self.meta_lr = meta_lr
        self.meta_optimizer = torch.optim.Adam(self.weights.parameters(), lr=self.meta_lr)

    def check_probs(self, pos, neg):
        pos_probs = pos[:, 0, (self.REL, self.NREL)].softmax(dim=-1)[:, 0]
        neg_probs = neg[:, 0, (self.REL, self.NREL)].softmax(dim=-1)[:, 0]

        self.logs['probs'].append(torch.mean((pos_probs > neg_probs).float()).item())

    def check_success_rate(self, loss):
        self.running_rate.append(torch.mean((loss < self.weights.eta.item()).float()).item())

    def prep_batch(self, batch):
        px, nx = batch

        px = self.tokenizer(px, padding=True, return_tensors='pt').input_ids.to(self.device)
        o_p = self.tokenizer(['true'] * len(px), padding=True, return_tensors='pt').input_ids[:, 0].view(-1, 1).to(self.device)

        nx = self.tokenizer(nx, padding=True, return_tensors='pt').input_ids.to(self.device)
        o_n = self.tokenizer(o_n, padding=True, return_tensors='pt').input_ids[:, 0].view(-1, 1).to(self.device)

        px = Variable(px, requires_grad=False)
        nx = Variable(nx, requires_grad=False)
        o_p = Variable(o_p, requires_grad=False)
        o_n = Variable(o_n, requires_grad=False)

        return px, nx, o_p, o_n
        

    def meta_loop(self, j):

        px, nx, op, on = self.prep_batch(self.train_loader.get_batch(j, self.difficulty))
 
        with torch.no_grad():
            plogits = self.model(input_ids=px, labels=op).logits
            nlogits = self.model(input_ids=nx, labels=on).logits

        loss = self.loss_fn(plogits, nlogits, op, on, self.weights)
       
        loss.backward()
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()

        self.weights.clamp()

        self.meta_scheduler.step()

        self.logs['loss']['meta'].append(loss.item())
        return loss.item()

    def main_loop(self, j):
        px, nx, o_p, o_n = self.prep_batch(self.train_loader.get_batch(j, self.difficulty))

        plogits = self.model(input_ids=px, labels=o_p).logits
        nlogits = self.model(input_ids=nx, labels=o_n).logits # Resolve dimensionality issues
        
        self.check_probs(plogits, nlogits)

        loss = self.loss_fn(plogits, nlogits, o_p, o_n)
        self.check_success_rate(loss)
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self, train_loader, total_steps, warmup_steps):
        torch.manual_seed(42)
        _logger = ir_datasets.log.easy()
        self.train_loader = train_loader   
        self.total_steps = total_steps
        self.difficulty = 0.0
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=warmup_steps // self.train_loader.batch_size if warmup_steps else (total_steps // 100), 
                                                         num_training_steps=total_steps)
        self.meta_scheduler = get_linear_schedule_with_warmup(self.meta_optimizer, 
                                                         num_warmup_steps=warmup_steps // self.train_loader.batch_size if warmup_steps else (total_steps // 100), 
                                                         num_training_steps=total_steps)
        
        start = time.time()
        
        with _logger.pbar_raw(desc=f'train', total=total_steps) as pbar:
            for i in range(total_steps//self.train_loader.batch_size):

                meta_loss = self.meta_loop(i)
                loss = self.main_loop(i)

                if wandb.run is not None:
                    wandb.log({'loss': loss, 
                               'meta_loss' : meta_loss, 
                               'lr': self.scheduler.get_last_lr()[0], 
                               'meta_lr' : self.meta_scheduler.get_last_lr()[0], 
                               'difficulty': self.difficulty, 
                               'success_rate' : self.running_rate[-1], 
                               'eta' : self.weights.eta.item(),
                               'probs' : self.logs['probs'][-1]})

                self.logs['loss']['main'].append(loss)
                self.logs['lr']['main'].append(self.scheduler.get_last_lr()[0])
                self.logs['lr']['meta'].append(self.meta_scheduler.get_last_lr()[0])
                self.logs['difficulty'].append(self.difficulty)
                self.logs['eta'].append(self.weights.eta.item())
              
                self.difficulty = self.weights.eta.item()
                pbar.set_postfix({'loss': np.mean(self.logs['loss']['main'])})
                pbar.update(self.train_loader.batch_size)

        end = time.time() - start

        self.logs['time'] = end
        if wandb.run is not None:
            wandb.log({'time': end})

        return self.logs