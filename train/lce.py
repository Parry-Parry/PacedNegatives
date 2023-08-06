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
        dataset = pd.read_json(f, orient='records', lines=True, dtype={'qid': str, 'doc_id_a': str, 'doc_id_b': list})
    corpus = irds.load(dataset_name)

    if sample: dataset = dataset.sample(frac=1).reset_index(drop=True)

    pairs = dataset[['qid', 'doc_id_a']].values.tolist()
    neg_idx = dataset['doc_id_b'].values

    dataset = LCEDataset(pairs, neg_idx, corpus, max)
    loader = LCELoader(dataset, batch_size, var, n)

    ## INIT MODEL ##

    init = lambda : T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    ## TRAIN ##

    trainer = LCEWrapper(eta
                         dataset_name, 
                         'monoT5', 
                         batch_size, 
                         init, 
                         tokenizer, 
                         lr, 
                         ignore_index=-100, 
                         max=max,)
    
    logs = trainer.train(loader, total_steps, warmup_steps=warmup_steps)

    trainer.model.save_pretrained(os.path.join(out_dir, 'model'))

    with open(os.path.join(out_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)