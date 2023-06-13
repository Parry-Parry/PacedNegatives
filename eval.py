import pyterrier as pt
pt.init()
import fire 
import os
from os.path import join
import torch 
from pyterrier_t5 import MonoT5ReRanker


def main(model_dir : str, out : str, corpus : str, eval_name : str):
    bm25 = pt.BatchRetrieve.from_dataset(corpus, 'terrier_stemmed_text', wmodel="BM25", metadata=['docno', 'text'])
    models = ['t5baseline16', 't5spl3']
    models = {store : bm25 > MonoT5ReRanker(model=join(model_dir, store, 'model')) for store in models}
    
    dataset = pt.get_dataset(eval_name)
    res = pt.Experiment(models.values(), dataset.get_topics(), dataset.get_qrels(), eval_metrics=["map", "nDCG@10", "mrr"], names = models.keys(), baseline = 0)
    qres = pt.Experiment(models.values(), dataset.get_topics(), dataset.get_qrels(), eval_metrics=["map", "nDCG@10", "mrr"], names = models.keys(), baseline = 0, perquery=True)
    os.makedirs(out, exist_ok=True)
    res.to_csv(join(out, "results.csv"))
    qres.to_csv(join(out, "perqueryresults.csv"))
    return "Success!"

if __name__ == '__main__':
    fire.Fire(main)