import pyterrier as pt
pt.init()
import fire 
import os
from os.path import join
import torch 
from pyterrier_t5 import MonoT5ReRanker


def main(model_dir : str, out : str, eval_name : str, baseline : str, model : str = None):  
    dataset = pt.get_dataset("irds:msmarco-passage/train/triples-small")
    bm25 = pt.BatchRetrieve(pt.get_dataset("msmarco_passage").get_index("terrier_stemmed_text"), wmodel="BM25")
    if not model:
        dirs = [f for f in os.listdir(model_dir) if os.path.isdir(join(model_dir, f))]
        models = {store : bm25 >> pt.text.get_text(dataset, "text") >> MonoT5ReRanker(model=join(model_dir, store, 'model')) for store in dirs}
        models = {'baseline' : bm25 >> pt.text.get_text(dataset, "text") >> MonoT5ReRanker(model=join(baseline, 'model')), **models}
    else:
        models = {'baseline' : bm25 >> pt.text.get_text(dataset, "text") >> MonoT5ReRanker(model=join(baseline, 'model')), 'model' : bm25 >> pt.text.get_text(dataset, "text") >> MonoT5ReRanker(model=join(model_dir, model, 'model'))}
    
    eval = pt.get_dataset(eval_name)
    res = pt.Experiment(list(models.values()), eval.get_topics(), eval.get_qrels(), eval_metrics=["map", "ndcg_cut_10", "recip_rank"], names = list(models.keys()), baseline = 0)
    qres = pt.Experiment(list(models.values()), eval.get_topics(), eval.get_qrels(), eval_metrics=["map", "ndcg_cut_10", "recip_rank"], names = list(models.keys()), perquery=True)
    os.makedirs(out, exist_ok=True)
    res.to_csv(join(out, "results.csv"))
    qres.to_csv(join(out, "perqueryresults.csv"))
    return "Success!"

if __name__ == '__main__':
    fire.Fire(main) 