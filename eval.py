import pyterrier as pt
pt.init()
import fire 
import os
from os.path import join
from pyterrier_t5 import MonoT5ReRanker
import pandas as pd


def main(model_dir : str, out : str, eval_name : str, baseline : str, model : str = None):  
    dataset = pt.get_dataset("irds:msmarco-passage/train/triples-small")
    bm25 = pt.BatchRetrieve(pt.get_dataset("msmarco_passage").get_index("terrier_stemmed_text"), wmodel="BM25")
    eval = pt.get_dataset(eval_name)
    baseline = bm25 >> pt.text.get_text(dataset, "text") >> MonoT5ReRanker(model=join(baseline, 'model'))
    os.makedirs(out, exist_ok=True)
    if not model:
        dirs = [f for f in os.listdir(model_dir) if os.path.isdir(join(model_dir, f))]
        tmp_res = []
        tmp_qres = []
        for i, store in enumerate(dirs):
            _model = bm25 >> pt.text.get_text(dataset, "text") >> MonoT5ReRanker(model=join(model_dir, store, 'model'))
            models = {'baseline' : baseline, store : _model}
            res = pt.Experiment(list(models.values()), eval.get_topics(), eval.get_qrels(), eval_metrics=["map", "ndcg_cut_10", "recip_rank"], names = list(models.keys()), baseline = 0)
            qres = pt.Experiment(list(models.values()), eval.get_topics(), eval.get_qrels(), eval_metrics=["map", "ndcg_cut_10", "recip_rank"], names = list(models.keys()), perquery=True)
            if i != 0:
                res = res.reset_index().drop(0)
                qres = qres.reset_index().drop(0)
            tmp_res.append(res)
            tmp_qres.append(qres)
            del _model
        
        res = pd.concat(tmp_res)
        qres = pd.concat(tmp_qres)
    else:
        models = {'baseline' : baseline, 'model' : bm25 >> pt.text.get_text(dataset, "text") >> MonoT5ReRanker(model=join(model_dir, model, 'model'))}
        res = pt.Experiment(list(models.values()), eval.get_topics(), eval.get_qrels(), eval_metrics=["map", "ndcg_cut_10", "recip_rank"], names = list(models.keys()), baseline = 0)
        qres = pt.Experiment(list(models.values()), eval.get_topics(), eval.get_qrels(), eval_metrics=["map", "ndcg_cut_10", "recip_rank"], names = list(models.keys()), perquery=True)
        
    res.to_csv(join(out, "results.csv"))
    qres.to_csv(join(out, "perqueryresults.csv"))
    return "Success!"

if __name__ == '__main__':
    fire.Fire(main) 