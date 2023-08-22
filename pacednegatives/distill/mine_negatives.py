import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
from pyterrier.model import add_ranks, split_df
from fire import Fire
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import ir_datasets as irds
import gc

def convert_to_dict(result):
    result = result.groupby('qid').apply(lambda x: dict(zip(x['docno'], zip(x['score'], x['rank'])))).to_dict()
    return result

class EnsembleScorer(pt.Transformer):
    DEFAULT = (0, 10000)
    def __init__(self, models, C=0) -> None:
        super().__init__()
        self.models = models
        self.C = C
    
    def get_fusion_scores(self, target_sets, qids):
        records = []
        if len(target_sets) == 1:
            target = target_sets[0]
            for qid in qids:
                for doc_id, (score, rank) in target[qid].items():
                    records.append({
                        'qid': str(qid),
                        'docno': str(doc_id),
                        'score': score,
                    })
            return pd.DataFrame.from_records(records)
        for qid in qids:
            all_sets = [set(target[qid].keys()) for target in target_sets]
            candidates = all_sets[0].union(*all_sets[1:])
            for candidate in candidates:
                for target in target_sets:
                    if candidate not in target[qid]:
                        target[qid][candidate] = self.DEFAULT
                scores = [1 / (self.C + target[qid][candidate][1] + 1) for target in target_sets]
                score = np.mean(scores)
                records.append({
                    'qid': str(qid),
                    'docno': str(candidate),
                    'score': score,
                })   
        return pd.DataFrame.from_records(records)

    def transform(self, inp):
        result_sets = []
        for model in tqdm(self.models, desc="Model Iter"):
            result_sets.append(model.transform(inp))
        sets = [convert_to_dict(res) for res in result_sets]
        qids = list(inp["qid"].unique())

        return add_ranks(self.get_fusion_scores(sets, qids))

def main(index_path : str, dataset_name : str, out_dir : str, subset : int = 100000, budget : int = 1000, batch_size : int = 1000, num_threads : int = 8):
    index = pt.IndexFactory.of(index_path, "terrier_stemmed")

    bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.k_1": 0.45, "bm25.b": 0.55, "bm25.k_3": 0.5})
    dph = pt.BatchRetrieve(index, wmodel="DPH")

    bo1 = pt.rewrite.Bo1QueryExpansion(index)
    kl = pt.rewrite.KLQueryExpansion(index)
    rm3 = pt.rewrite.RM3(index)

    models = [
        (bm25 >> bo1 >> bm25 % budget).parallel(num_threads),
        (bm25 >> kl >> bm25 % budget).parallel(num_threads),
        (bm25 >> rm3 >> bm25 % budget).parallel(num_threads),
        (dph >> bo1 >> dph % budget).parallel(num_threads),
        (dph >> kl >> dph % budget).parallel(num_threads),
    ]

    scorer = EnsembleScorer(models, C=0.0)

    dataset = irds.load(dataset_name)
    queries = pd.DataFrame(dataset.queries_iter()).set_index('qid')['text'].to_dict()
    train = pd.DataFrame(dataset.docpairs_iter()).drop(['doc_id_b'], axis=1).rename(columns={'query': 'qid',})
    train = dataset.sample(n=subset) 

    train['query'] = train['qid'].apply(lambda x : queries[x])
    del queries
    del dataset
    gc.collect()

    new_set = []

    for subset in tqdm(split_df(train, batch_size), desc="Total Batched Iter"):
        new = subset.copy()
        res = scorer.transform(subset).drop(['score', 'rank'], axis=1)
        new['doc_id_b'] = res.apply(lambda x : x[x.qid==subset.qid]['docno'].iloc[:1000].sample(n=1))
        new_set.append(new)

    new_set = pd.concat(new_set)
    new_set.to_csv(out_dir, sep='\t', index=False)

    return "Done!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)








