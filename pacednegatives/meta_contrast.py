import time
import torch 
import numpy as np
import ir_datasets
import wandb
from torch.autograd import grad

from transformers import get_linear_schedule_with_warmup
from pacednegatives.pairwrapper import PacedWrapper
from pacednegatives.weights import EtaWeights

def interpolate_scalar(start_value, end_value, num_steps):
    step_size = (end_value - start_value) / num_steps
    
    def get_interpolated_value(step):
        if step > num_steps:
            return end_value
        return start_value + step * step_size
    
    return get_interpolated_value

class MetaContrastWrapper(PacedWrapper):
    def __init__(self, 
                 eta,
                 min_eta,
                 max_eta,
                 rate_check,
                 threshold,
                 dataset, 
                 model_name, 
                 batch_size, 
                 model_init, 
                 tokenizer, 
                 lr, 
                 meta_lr,
                 ignore_index) -> None:
        super().__init__(dataset, model_name, batch_size, model_init, tokenizer, lr, ignore_index)

        self.weights = EtaWeights(eta, device=self.device, min=np.log(min_eta), max=max_eta)
        self.logs['eta'] = []
        self.logs['loss'] = {'main': [], 'meta' : []}
        self.logs['difficulty'] = []
        self.logs['lr'] = {'main': [], 'meta' : []}
        self.logs['success_rate'] = []

        self.running_rate = []
        self.rate_check = rate_check
        self.threshold = threshold

        self.meta_lr = meta_lr
        self.meta_optimizer = torch.optim.Adam(self.weights.parameters(), lr=self.meta_lr)

    def check_success_rate(self, loss):
        self.running_rate.append(torch.mean((loss < self.weights.eta.item()).float()).item())

    def meta_loop(self, j):

        px, nx, o_p, o_n = self.prep_batch(self.train_loader.get_batch(j, self.difficulty))
 
        '''
        with torch.no_grad():
            plogits = self.meta_model(input_ids=px, labels=o_p).logits
            nlogits = self.meta_model(input_ids=nx, labels=o_n).logits
        '''
        with torch.no_grad():
            plogits = self.model(input_ids=px, labels=o_p).logits
            nlogits = self.model(input_ids=nx, labels=o_n).logits

        pce = self.loss_fn(plogits.view(-1, plogits.size(-1)), o_p.view(-1))
        nce = self.loss_fn(nlogits.view(-1, nlogits.size(-1)), o_n.view(-1))
      
        ce = torch.div(pce+nce, 2)
        v = self.weights.forward(loss=ce)
        #weighted_ce = torch.mean(pce * v) + torch.mean(nce * v) - torch.sum(v)
        weighted_ce = torch.mean(pce * v) + torch.mean(nce * v) 

        weighted_ce.backward()
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()

        self.weights.clamp()
        #grads = grad(weighted_ce, (self.weights.eta, ), create_graph=True, retain_graph=True)
        #self.weights.eta = self.weights.clamp(self.weights.eta - self.meta_scheduler.get_last_lr()[0] * grads[0])
        #del grads

        self.meta_scheduler.step()

        self.logs['loss']['meta'].append(weighted_ce.item())
        return weighted_ce.item()

    def main_loop(self, j):
        px, nx, o_p, o_n = self.prep_batch(self.train_loader.get_batch(j, self.difficulty))

        logits_p = self.model(input_ids=px, labels=o_p).logits
        pce = self.loss_fn(logits_p.view(-1, logits_p.size(-1)), o_p.view(-1))

        logits_n = self.model(input_ids=nx, labels=o_n).logits
        nce = self.loss_fn(logits_n.view(-1, logits_n.size(-1)), o_n.view(-1))
        
        ce = torch.div(pce+nce, 2)
        # v = self.weights.no_grad(ce, self.weights.eta)
        # loss = torch.mean(pce * v) + torch.mean(nce * v)
        loss = torch.mean(pce) + torch.mean(nce)

        self.check_success_rate(ce)
        
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
        #mask_schedule = interpolate_scalar(1.0, 0.0, warmup_steps // self.train_loader.batch_size if warmup_steps else (total_steps // 100))
        
        start = time.time()
        
        with _logger.pbar_raw(desc=f'train', total=total_steps) as pbar:
            for i in range(total_steps//self.train_loader.batch_size):
                #self.weights.set_mask(mask_schedule(i))
                meta_loss = self.meta_loop(i)
                loss = self.main_loop(i)

                if wandb.run is not None:
                    wandb.log({'loss': loss, 'meta_loss' : meta_loss, 'lr': self.scheduler.get_last_lr()[0], 'meta_lr' : self.meta_scheduler.get_last_lr()[0], 'difficulty': self.difficulty, 'success_rate' : self.running_rate[-1], 'eta' : self.weights.eta.item()})

                self.logs['loss']['main'].append(loss)
                self.logs['lr']['main'].append(self.scheduler.get_last_lr()[0])
                self.logs['lr']['meta'].append(self.meta_scheduler.get_last_lr()[0])
                self.logs['difficulty'].append(self.difficulty)
                self.logs['eta'].append(self.weights.eta.item())

               
                if i % self.rate_check == 0:
                    success_rate = np.mean(self.running_rate)
                    if success_rate > self.threshold:
                        self.difficulty = min(1., (self.difficulty + (1 / train_loader.dataset.n_neg)))
                    self.running_rate = []

                pbar.set_postfix({'loss': np.mean(self.logs['loss']['main'])})
                pbar.update(self.train_loader.batch_size)

        end = time.time() - start

        self.logs['time'] = end
        if wandb.run is not None:
            wandb.log({'time': end})

        return self.logs