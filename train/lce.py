from fire import Fire
from pacednegatives.lce import LCEWrapper
from pacednegatives.dataloader import LCEDataset, LCELoader
import os
import json
import pandas as pd
import ir_datasets as irds
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
import wandb
import numpy as np
from typing import List
from torch.utils.data import DataLoader

def main(
        data : str, 
        dataset_name : str, 
        out_dir : str, 
        total_steps : int = 100000, 
        eta : float = 0.0,
        batch_size : int = 16, 
        lr : float = 0.001, 
        var : float = 0.01,
        n : int = 2,
        max=True,
        warmup_steps=0,
        sample=False,
        use_mean=True,
        wandb_project=None,):

    os.makedirs(out_dir, exist_ok=True)

    if wandb_project is not None:
        wandb.init(project=wandb_project, config={
                'variant': data.split('/')[-1],
                'dataset': dataset_name,
                'total_steps': total_steps,
                'eta': eta,
                'batch_size': batch_size,
                'lr': lr,
                'max': max,
                'warmup_steps': warmup_steps,
            })

    ## INIT DATA ##

    with open(data, 'r') as f:
        dataset = pd.read_json(f, orient='records', dtype={'query_id': str, 'doc_id_a': str, 'doc_id_b': List[str]})
    corpus = irds.load(dataset_name)

    if sample: dataset = dataset.sample(frac=1).reset_index(drop=True)

    pairs = dataset[['query_id', 'doc_id_a']].values.tolist()
    neg_idx = dataset['doc_id_b'].values

    dataset = LCEDataset(pairs, neg_idx, corpus, max)
    loader = LCELoader(dataset, batch_size, var, n, min=0.+1e-10, max=1.0-1e-10)

    ## INIT MODEL ##

    init = lambda : T5ForConditionalGeneration.from_pretrained('t5-base', device_map='auto')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    ## TRAIN ##

    trainer = LCEWrapper(eta,
                         dataset_name, 
                         'monoT5', 
                         batch_size, 
                         init, 
                         tokenizer, 
                         lr, 
                         lr,
                         ignore_index=-100, 
                         use_mean=use_mean,
                         )
    
    logs = trainer.train(loader, total_steps, warmup_steps=warmup_steps)
    trainer.accelerator.save_model(trainer.model, os.path.join(out_dir, 'model'))

    with open(os.path.join(out_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)