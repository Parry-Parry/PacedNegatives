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
        for model in tqdm(self.models):
            result_sets.append(model.transform(inp))
        sets = [convert_to_dict(res) for res in result_sets]
        qids = list(inp["qid"].unique())

        return add_ranks(self.get_fusion_scores(sets, qids))

def main(index_path : str, dataset_name : str, out_dir : str, subset : int = 100000, budget : int = 1000, batch_size : int = 1000):
    index = pt.IndexFactory.of(index_path)

    bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.k_1": 0.45, "bm25.b": 0.55, "bm25.k_3": 0.5})
    dph = pt.BatchRetrieve(index, wmodel="DPH")

    bo1 = pt.rewrite.Bo1QueryExpansion(index)
    kl = pt.rewrite.KLQueryExpansion(index)
    rm3 = pt.rewrite.RM3(index)

    models = [
        bm25 >> bo1 >> bm25 % budget,
        bm25 >> kl >> bm25 % budget,
        bm25 >> rm3 >> bm25 % budget,
        dph >> bo1 >> dph % budget,
        dph >> kl >> dph % budget,
    ]

    scorer = EnsembleScorer(models, C=0.0)

    dataset = irds.load(dataset_name)
    train = pd.DataFrame(dataset.scoredocs_iter())
    train = dataset.sample(n=subset) 

    new_set = []

    for subset in split_df(train, batch_size):
        new = subset.copy()
        res = scorer.transform(subset)
        new['doc_id_b'] = subset.apply(lambda x : res[res.qid==subset.qid].sample(n=1))
        new_set.append(new)

    new_set = pd.concat(new_set).rename(columns={'docno': 'doc_id_a'})




    topics.to_csv(join(out_dir, 'topics.tsv'), sep="\t", index=False, header=True)

    return "Done!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)








