from fire import Fire
from pacednegatives.meta_contrast import MetaContrastWrapper
from pacednegatives.dataloader import TripletDataset, LevelLoader
import os
import json
import pandas as pd
import ir_datasets as irds
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
import numpy as np

def main(
        data : str, 
        dataset_name, 
        out_dir : str, 
        batch_size : int = 32, 
        lr : float = 0.001, 
        max=True, 
        eta=-np.log(0.5)*0.5, 
        min_eta=0.01, 
        max_eta=15,
        threshold=0.5,
        rate_check=1000,
        training_steps=100000,
        warmup_steps=2500):

    os.makedirs(out_dir, exist_ok=True)

    ## INIT DATA ##

    dataset = pd.read_json(data, orient='records', lines=True)
    corpus = irds.load(dataset_name)

    pairs = dataset[['query_id', 'doc_id_a']].values.tolist()
    neg_idx = dataset['doc_id_b'].values.to_numpy()

    dataset = TripletDataset(pairs, neg_idx, corpus, max)
    loader = LevelLoader(dataset, batch_size)

    ## INIT MODEL ##

    init = lambda : T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    ## TRAIN ##

    trainer = MetaContrastWrapper(eta, min_eta, max_eta, rate_check, threshold, dataset_name, 'monoT5', batch_size, init, tokenizer, lr, -100)
    logs = trainer.train(loader, training_steps, warmup_steps=warmup_steps)

    trainer.model.save_pretrained(os.path.join(out_dir, 'model'))

    with open(os.path.join(out_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)