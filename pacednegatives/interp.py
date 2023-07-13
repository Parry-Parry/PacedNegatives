import time
import torch 
import numpy as np
import ir_datasets
import wandb

from transformers import get_linear_schedule_with_warmup
from pacednegatives.pairwrapper import PacedWrapper

def interpolate_scalar(start_value, end_value, num_steps):
    step_size = (end_value - start_value) / num_steps
    
    def get_interpolated_value(step):
        if step > num_steps:
            return end_value
        return start_value + step * step_size
    
    return get_interpolated_value
    
class InterpWrapper(PacedWrapper):
    def __init__(self, 
                 dataset, 
                 model_name, 
                 batch_size, 
                 model_init, 
                 tokenizer, 
                 lr, 
                 ignore_index,
                 start_difficulty,
                 max_difficulty,
                 interpolate_steps) -> None:
        super().__init__(dataset, model_name, batch_size, model_init, tokenizer, lr, ignore_index)

        self.difficulty = start_difficulty
        self.start_difficulty = start_difficulty
        self.max_difficulty = max_difficulty
        self.interpolate_steps = interpolate_steps

        self.interpolate_difficulty = interpolate_scalar(start_difficulty, max_difficulty, interpolate_steps)

        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

        self.logs['difficulty'] = {'train': []}
        self.logs['lr'] = []

    def main_loop(self, j):
        px, nx, o_p, o_n = self.prep_batch(self.train_loader.get_batch(j, self.difficulty))

        logits_p = self.model(input_ids=px, labels=o_p).logits
        ce_p = self.loss_fn(logits_p.view(-1, logits_p.size(-1)), o_p.view(-1))

        logits_n = self.model(input_ids=nx, labels=o_n).logits
        ce_n = self.loss_fn(logits_n.view(-1, logits_n.size(-1)), o_n.view(-1))

        loss = torch.mean(ce_p) + torch.mean(ce_n)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train(self, train_loader, total_steps, warmup_steps):
        torch.manual_seed(42)
        _logger = ir_datasets.log.easy()
        self.train_loader = train_loader   
        self.total_steps = total_steps
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=warmup_steps // self.train_loader.batch_size if warmup_steps else (total_steps // 100), 
                                                         num_training_steps=total_steps)
        
        start = time.time()

        with _logger.pbar_raw(desc=f'train', total=total_steps) as pbar:
            for i in range(total_steps//self.train_loader.batch_size):
        
                loss = self.main_loop(i)

                if wandb.run is not None:
                    wandb.log({'loss': loss, 'success_rate' : self.success_rate[-1],'lr': self.scheduler.get_last_lr()[0], 'difficulty': self.difficulty})

                self.difficulty = self.interpolate_difficulty(i*self.train_loader.batch_size)

                self.logs['loss']['train'].append(loss)
                self.logs['lr'].append(self.scheduler.get_last_lr()[0])
                self.logs['difficulty']['train'].append(self.difficulty)

                pbar.update(self.train_loader.batch_size)

        end = time.time() - start

        self.logs['time'] = end
        if wandb.run is not None:
            wandb.log({'time': end})

        return self.logs