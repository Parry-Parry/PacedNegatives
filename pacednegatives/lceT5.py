from collections import OrderedDict
import itertools
from pacednegatives.dataloader import LCEDataset
import lightning.pytorch as pl
from transformers import get_linear_schedule_with_warmup, AdamW, T5ForConditionalGeneration, T5Tokenizer
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import ir_datasets as irds
import pandas as pd
from typing import List, Any

class LCEDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = "path/to/dir", 
                 corpus: str = "msmarco-passage", 
                 tokenizer : Any = None,
                 batch_size: int = 32, 
                 shuffle=False, 
                 use_max=False, 
                 var=0.01, 
                 n=2):
        super().__init__()
        self.data_dir = data_dir
        self.corpus = irds.load(corpus)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_max = use_max
        self.var = var
        self.n = n

        self.weight = 0. + 1e-10
    
    def collate(batch):
       pos = [x['pos'] for x in batch]
       neg = [x['neg'] for x in batch]  

        

    def setup(self, stage: str=None):
        with open(self.data_dir, 'r') as f:
            dataset = pd.read_json(f, orient='records', dtype={'query_id': str, 'doc_id_a': str, 'doc_id_b': List[str]})
        if self.shuffle: dataset = dataset.sample(frac=1).reset_index(drop=True)

        self.pairs = dataset[['query_id', 'doc_id_a']].values.tolist()
        self.neg_idx = dataset['doc_id_b'].values
        self.dataset = LCEDataset(self.pairs, self.neg_idx, self.corpus, self.tokenizer, self.batch_size, var=self.var, n=self.n, min=0.+1e-10, max=1.0-1e-10, use_max=self.use_max)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4)

def batch_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def concatenate(*lists):
    return list(itertools.chain(*lists))

class ChangeDifficulty(pl.Callback):
    def __init__(self):
        pass 
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        trainer.train_dataloader.dataset.weight = min(1-1e-10, pl_module.weights.eta.item())

class LCEWeights(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor([hparams.eta]), requires_grad=True)
        self.register_parameter('eta_value', self.eta)

    def weighting(self, x, y):
        return (-x/y)
    
    def forward(self, loss):
        weight = torch.zeros(loss.size()).to(loss.device)

        for i in range(len(loss)):
            if loss[i] > self.eta:
                weight[i] = loss[i] * torch.zeros(1, requires_grad=True).to(loss.device) * self.eta
            else:
                weight[i] = self.weighting(loss[i], self.eta)
        return weight

class LCEModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        for key in hparams.keys():
            self.hparams[key]=hparams[key]
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name)
        self.weights = LCEWeights(self.hparams)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.hparams.ignore_index, reduction='none')
        self.automatic_optimization = False
        self.save_hyperparameters()

    def create_y(self, x, token='false'):
        y = self.tokenizer([token] * len(x), max_length=512, return_tensors='pt').input_ids[:, 0].view(-1, 1)
        return y

    def prep_batch(self, batch):
        p, tmp = batch
        n = []
        for _n in tmp: 
            n.extend(list(_n))

        p = self.tokenizer(p, max_length=512, return_tensors='pt', padding='max_length', truncation=True)
        n = self.tokenizer(n, max_length=512, return_tensors='pt', padding='max_length', truncation=True)

        p['labels'] = self.create_y(p, token='true')
        n['labels'] = self.create_y(n, token='false')

        p = p.to(self.device)
        n = n.to(self.device)

        return p, n
    
    def pair_loss(self, p, n, op, on):
        pce = self.loss_fn(p.view(-1, n.size(-1)), op.view(-1))
        nce = self.loss_fn(n.view(-1, n.size(-1)), on.view(-1))
        nce = nce.view(-1, n.size(-2))
        nce = torch.mean(nce, dim=1) if self.hparams.use_mean else torch.sum(nce, dim=1)
        ce = pce + nce

        return ce

    def training_step(self, batch, batch_nb):
        p, n = self.prep_batch(batch)

        meta_opt, opt = self.optimizers()
        meta_scheduler, scheduler = self.lr_schedulers()

        with torch.no_grad():
            plogits = self.model(**p).logits
            nlogits = self.model(**n).logits
        nlogits = nlogits.view(-1, self.hparams.n, nlogits.size(-1)) # Resolve dimensionality issues
        loss = self.pair_loss(plogits, nlogits, p['labels'], n['labels'])

        weights = self.weights(loss)
        meta_loss = weights * loss

        meta_opt.zero_grad()
        self.manual_backward(meta_loss.mean())
        meta_opt.step()
        meta_scheduler.step()

        self.log('avg_weight', weights.mean())
        self.log('meta_loss', loss.mean())

        plogits = self.model(**p).logits
        nlogits = self.model(**n).logits
        nlogits = nlogits.view(-1, self.hparams.n, nlogits.size(-1)) # Resolve dimensionality issues
        main_loss = self.pair_loss(plogits, nlogits, p['labels'], n['labels'])
        
        opt.zero_grad()
        self.manual_backward(main_loss.mean())
        opt.step()
        scheduler.step()

        tqdm_dict = {'meta_loss': loss.mean(), 'avg_weight': weights.mean(), 'main_loss': loss.mean()}
        output = OrderedDict({
            'main_loss': main_loss.mean(),
            'meta_loss': meta_loss.mean(),
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        self.log('main_loss', loss.mean())
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