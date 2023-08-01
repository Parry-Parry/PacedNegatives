import time
import torch 
import numpy as np
import ir_datasets
import wandb

from transformers import get_linear_schedule_with_warmup
from pacednegatives.pairwrapper import PacedWrapper


class LevelWrapper(PacedWrapper):
    def __init__(self, 
                 dataset, 
                 model_name, 
                 batch_size, 
                 model_init, 
                 tokenizer, 
                 lr, 
                 ignore_index,
                 success_threshold,
                 heuristic_step_check) -> None:
        super().__init__(dataset, model_name, batch_size, model_init, tokenizer, lr, ignore_index)

        self.difficulty = 0.0
        self.success_threshold = success_threshold
        self.heuristic_step_check = heuristic_step_check
        self.success_rate = []

        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

        self.logs['difficulty'] = {'train': []}
        self.logs['lr'] = []

    def check_success(self, pos, neg):
        pos_probs = pos[:, 0, (self.REL, self.NREL)].softmax(dim=-1)[:, 0]
        neg_probs = neg[:, 0, (self.REL, self.NREL)].softmax(dim=-1)[:, 0]

        self.success_rate.append(torch.mean((pos_probs > neg_probs).float()).item())

    def main_loop(self, j):
        px, nx, o_p, o_n = self.prep_batch(self.train_loader.get_batch(j, self.difficulty))

        logits_p = self.model(input_ids=px, labels=o_p).logits
        ce_p = self.loss_fn(logits_p.view(-1, logits_p.size(-1)), o_p.view(-1))

        logits_n = self.model(input_ids=nx, labels=o_n).logits
        ce_n = self.loss_fn(logits_n.view(-1, logits_n.size(-1)), o_n.view(-1))

        self.check_success(logits_p, logits_n)

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
                                                         num_warmup_steps=warmup_steps if warmup_steps else total_steps // 100, 
                                                         num_training_steps=total_steps)
        
        start = time.time()

        with _logger.pbar_raw(desc=f'train', total=total_steps) as pbar:
            for i in range(total_steps//self.train_loader.batch_size):
                if i % self.heuristic_step_check == 0 and len(self.success_rate) > 0:
                    agg_success_rate = np.mean(self.success_rate)
                    if agg_success_rate > self.success_threshold:
                        self.difficulty = min(1.0, self.difficulty + 1 / train_loader.dataset.n_neg)
                    self.success_rate = []

                loss = self.main_loop(i)

                if wandb.run is not None:
                    wandb.log({'loss': loss, 'success_rate' : self.success_rate[-1],'lr': self.scheduler.get_last_lr()[0], 'difficulty': self.difficulty})

                self.logs['loss']['train'].append(loss)
                self.logs['success_rate'] = self.success_rate[-1]
                self.logs['lr'].append(self.scheduler.get_last_lr()[0])
                self.logs['difficulty']['train'].append(self.difficulty)

                pbar.update(self.train_loader.batch_size)

        end = time.time() - start

        self.logs['time'] = end
        if wandb.run is not None:
            wandb.log({'time': end})

        return self.logs