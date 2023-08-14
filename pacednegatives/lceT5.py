from collections import OrderedDict
from pacednegatives.dataloader import LCEDataset
import lightning.pytorch as pl
from transformers import get_linear_schedule_with_warmup, AdamW
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import ir_datasets as irds
import pandas as pd
from typing import List

class LCEDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", corpus: str = "msmarco-passage", batch_size: int = 32, shuffle=False, use_max=False, var=0.01, n=2):
        super().__init__()
        self.data_dir = data_dir
        self.corpus = irds.load(corpus)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_max = use_max
        self.var = var
        self.n = n

        self.weight = 0. + 1e-10
    
    def collate(batch):
        batch = list(zip(*batch))

    def setup(self, stage: str=None):
        with open(self.data_dir, 'r') as f:
            dataset = pd.read_json(f, orient='records', dtype={'query_id': str, 'doc_id_a': str, 'doc_id_b': List[str]})
        if self.shuffle: dataset = dataset.sample(frac=1).reset_index(drop=True)

        self.pairs = dataset[['query_id', 'doc_id_a']].values.tolist()
        self.neg_idx = dataset['doc_id_b'].values
        self.dataset = LCEDataset(self.pairs, self.neg_idx, self.corpus, self.batch_size, var=self.var, n=self.n, min=0.+1e-10, max=1.0-1e-10, use_max=self.use_max)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate)

def batch_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class ChangeDifficulty(pl.Callback):
    def __init__(self):
        pass 
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        trainer.train_dataloader.dataset.weight = min(1-1e-10, pl_module.weights.eta.item())

class LCET5(pl.LightningModule):
    def __init__(self, hparams):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name)

    def forward(self, **kwargs):
        return self.model(**kwargs)

class LCEWeights(pl.LightningModule):
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

class LCEModel(pl.LightningModule):
    def __init__(self, hparams):
        self.hparams = hparams
        self.model = LCET5(hparams)
        self.weights = LCEWeights(hparams)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=hparams.ignore_index, reduction='none')
        self.y_neg = self.create_y(torch.ones(hparams.batch_size), token='false')

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
                plogits = self.model(input_ids=p, labels=op).logits
                nlogits = []
                for _batch in batch_iter(n, n=int(self.hparams.batch_size)):
                    nlogits.append(self.model(input_ids=_batch, labels=self.create_y(_batch, token='false').to(self.model.device)).logits)
            nlogits = torch.cat(nlogits, dim=0).view(-1, self.hparams.n, nlogits[0].size(-1)) # Resolve dimensionality issues
            loss = self.pair_loss(plogits, nlogits, op, on)
            weights = self.weights(loss)
            loss = weights * loss

            tqdm_dict = {'meta_loss': loss.mean(), 'avg_weight': weights.mean()}
            output = OrderedDict({
                'meta_loss': loss.mean(),
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        elif optimizer_idx == 1:    
            plogits = self.model(input_ids=p, labels=op).logits
            nlogits = []
            for _batch in batch_iter(n, n=int(self.hparams.batch_size)):
                nlogits.append(self.model(input_ids=_batch, labels=self.create_y(_batch, token='false').to(self.model.device)).logits)
            nlogits = torch.cat(nlogits, dim=0).view(-1, self.hparams.n, nlogits[0].size(-1)) # Resolve dimensionality issues
            loss = self.pair_loss(plogits, nlogits, op, on)

            tqdm_dict = {'main_loss': loss.mean()}
            output = OrderedDict({
                'main_loss': loss.mean(),
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

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