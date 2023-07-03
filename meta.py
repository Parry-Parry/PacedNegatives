from fire import Fire
from pacednegatives.pairwrapper import MetaWrapper
from pacednegatives.dataloader import TripletDataset, TripletLoader
import os
import json
import pandas as pd
import ir_datasets as irds
from transformers import T5ForConditionalGeneration, T5Tokenizer

def main(
        data : str, 
        dataset_name, 
        out_dir : str, 
        epochs : int = 10, 
        batch_size : int = 32, 
        lr : float = 0.001, 
        max=True, 
        eta=0.1, 
        min_eta=0.01, 
        max_eta=15,
        warmup_steps=0):

    os.makedirs(out_dir, exist_ok=True)

    ## INIT DATA ##

    dataset = pd.DataFrame.from_json(data, orient='records', lines=True)
    corpus = irds.load(dataset_name)

    pairs = dataset[['query_id', 'doc_id_a']].values.tolist()
    neg_idx = dataset['doc_id_b'].values.to_numpy()

    dataset = TripletDataset(pairs, neg_idx, corpus, max)
    loader = TripletLoader(dataset, batch_size)

    ## INIT MODEL ##

    init = lambda : T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    ## TRAIN ##

    trainer = MetaWrapper(eta, min_eta, max_eta, dataset_name, 'monoT5', batch_size, init, tokenizer, lr, -100)
    logs = trainer.train(loader, epochs, warmup_steps=warmup_steps)

    trainer.model.save_pretrained(os.path.join(out_dir, 'model'))

    with open(os.path.join(out_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f)

if __name__ == '__main__':
    Fire(main)