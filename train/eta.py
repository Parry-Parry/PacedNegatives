from fire import Fire
from pacednegatives.eta_bound import EtaWrapper
from pacednegatives.dataloader import TripletDataset, LevelLoader
import os
import json
import pandas as pd
import ir_datasets as irds
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
import numpy as np
import wandb

def main(
        data : str, 
        dataset_name, 
        out_dir : str, 
        batch_size : int = 32, 
        lr : float = 0.001, 
        meta_lr : float = 0.001,
        max=False, 
        sample : bool = False,
        eta = 0.5,
        training_steps=100000,
        warmup_steps=2500, 
        wandb_project=None,):

    os.makedirs(out_dir, exist_ok=True)

    if wandb_project is not None:
        wandb.init(project=wandb_project, config={
                'variant': data.split('/')[-1],
                'dataset': dataset_name,
                'total_steps': training_steps,
                'batch_size': batch_size,
                'lr': lr,
                'max': max,
                'warmup_steps': warmup_steps,
                'eta': eta,
            })

    ## INIT DATA ##

    with open(data, 'r') as f:
        dataset = pd.read_json(f, orient='records', lines=True, dtype={'query_id': str, 'doc_id_a': str, 'doc_id_b': list})
    corpus = irds.load(dataset_name)

    if sample: dataset = dataset.sample(frac=1).reset_index(drop=True)

    pairs = dataset[['query_id', 'doc_id_a']].values.tolist()
    neg_idx = dataset['doc_id_b'].values

    dataset = TripletDataset(pairs, neg_idx, corpus, max)
    loader = LevelLoader(dataset, batch_size)

    ## INIT MODEL ##

    init = lambda : T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    ## TRAIN ##

    trainer = EtaWrapper(eta,           
                        dataset_name, 
                        'monoT5', 
                        batch_size, 
                        init, 
                        tokenizer, 
                        lr, 
                        meta_lr,
                        -100)
    
    logs = trainer.train(loader, training_steps, warmup_steps=warmup_steps)

    trainer.model.save_pretrained(os.path.join(out_dir, 'model'))

    with open(os.path.join(out_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)