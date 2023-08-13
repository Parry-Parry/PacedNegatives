from collections import OrderedDict
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup, AdamW
import torch 
import torch.nn as nn

def batch_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class lceT5(pl.LightningModule):
    def __init__(self, hparams):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.hparams = hparams
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name)

    def forward(self, **kwargs):
        return self.model(**kwargs)

class lceWeights(pl.LightningModule):
    def __init__(self, hparams):
        self.eta = nn.Parameter(torch.tensor([hparams.eta]), requires_grad=True)
        self.register_parameter('eta_value', self.eta)

    def weighting(self, x, y):
        return (-x/y)
    
    def forward(self, loss):
        weight = torch.zeros(loss.size())

        for i in range(len(loss)):
            if loss[i] > self.eta:
                weight[i] = loss[i] * torch.zeros(1, requires_grad=True) * self.eta
            else:
                weight[i] = self.weighting(loss[i], self.eta)
        return weight

class lceModel(pl.LightningModule):
    def __init__(self, hparams):
        self.hparams = hparams
        self.model = lceT5(hparams)
        self.weights = lceWeights(hparams)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=hparams.ignore_index, reduction='none')
    
    def create_y(self, x, token='false'):
        y = self.model.tokenizer([token] * len(x), padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids[:, 0].view(-1, 1)
        return y

    def prep_batch(self, batch):
        p, n = batch

        p = self.model.tokenizer(p, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids
        n = self.model.tokenizer(n, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids

        op = self.create_y(p, token='true')
        on = self.create_y(n, token='false')

        return p, n, op, on
    
    def pair_loss(self, p, n, op, on):
        pce = self.loss_fn(p.view(-1, n.size(-1)), op.view(-1))
        nce = self.loss_fn(n.view(-1, n.size(-1)), on.view(-1))
        nce = nce.view(-1, n.size(-2))
        nce = torch.mean(nce, dim=1) if self.hparams.use_mean else torch.sum(nce, dim=1)
        ce = pce + nce

        return ce

    def training_step(self, batch, batch_nb, optimizer_idx):
        p, n, op, on = self.prep_batch(batch)

        if optimizer_idx == 0:
            with torch.no_grad():
                ploss = self.model(p, op).logits
                nlogits = []
                for _batch in batch_iter(n, n=int(self.hparams.batch_size)):
                    nlogits.append(self.model(input_ids=_batch, labels=self.y_neg).logits)
            nlogits = torch.cat(nlogits, dim=0).view(-1, self.train_loader.n, nlogits[0].size(-1)) # Resolve dimensionality issues
            loss = self.pair_loss(ploss, nlogits, op, on)
            weights = self.weights(loss)
            loss = weights * loss

            tqdm_dict = {'loss': loss.mean(), 'avg_weight': weights.mean()}
            output = OrderedDict({
                'loss': loss.mean(),
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        elif optimizer_idx == 1:    
            plogits = self.model(n, on).logits
            nlogits = []
            for _batch in batch_iter(n, n=int(self.hparams.batch_size)):
                nlogits.append(self.model(input_ids=_batch, labels=self.y_neg).logits)
            nlogits = torch.cat(nlogits, dim=0).view(-1, self.train_loader.n, nlogits[0].size(-1)) # Resolve dimensionality issues
            loss = self.pair_loss(plogits, nlogits, op, on)

    def configure_optimizers(self):
        meta_opt = AdamW(self.weights.parameters(), lr=self.hparams.meta_lr)
        opt = AdamW(self.model.parameters(), lr=self.hparams.lr)

        meta_sched = get_linear_schedule_with_warmup(meta_opt, 
                                                num_warmup_steps=self.hparams.warmup_steps, 
                                                num_training_steps=self.hparams.total_steps)
        sched = get_linear_schedule_with_warmup(opt, 
                                                num_warmup_steps=self.hparams.warmup_steps, 
                                                num_training_steps=self.hparams.total_steps)
        
        
        return [meta_opt, opt], [meta_sched, sched]