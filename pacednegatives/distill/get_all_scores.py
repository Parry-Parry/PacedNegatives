from collections import defaultdict
import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
from pyterrier.model import split_df
from fire import Fire
import pandas as pd
from tqdm import tqdm
import logging
import json
import re
from math import ceil

def convert_to_dict(result):
    result = result.groupby('qid').apply(lambda x: dict(zip(x['docno'], zip(x['score'], x['rank'])))).to_dict()
    return result

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def main(triples_path : str,
         index_path : str,
         out_path : str,
         batch_size : int = 1000,) -> str:
    
    index = pt.get_dataset(index_path).get_index("terrier_stemmed")
    index = pt.IndexFactory.of(index, memory=True)
    triples = pd.read_csv(triples_path, sep="\t", dtype={"qid":str, "doc_id_a":str, "query":str, "doc_id_b":str}, index_col=False)

    bm25 = pt.BatchRetrieve(index, controls={"bm25.k_1": 0.45, "bm25.b": 0.55, "bm25.k_3": 0.5})
    dph = pt.BatchRetrieve(index, wmodel="DPH")

    bo1 = pt.rewrite.Bo1QueryExpansion(index)
    kl = pt.rewrite.KLQueryExpansion(index)
    rm3 = pt.rewrite.RM3(index)

    models = [  
        (bm25 >> bo1 >> bm25 % 1000),
        (bm25 >> kl >> bm25 % 1000),
        (bm25 >> rm3 >> bm25 % 1000),
        (dph >> bo1 >> dph % 1000),
        (dph >> kl >> dph % 1000),
    ]

    def pivot_batch(batch):
        records = []
        for row in batch.itertuples():
            records.extend([{
                'qid': row.qid,
                'query': row.query,
                'docno': row.doc_id_a,
                },
                {
                'qid': row.qid,
                'query': row.query,
                'docno': row.doc_id_b,
                }
                ])
        return pd.DataFrame.from_records(records)

    def convert_to_dict(result):
        lookup = defaultdict(lambda : defaultdict(lambda : 0))
        for row in result.itertuples():
            lookup[row.qid][row.docno] = row.score
        return lookup
    
    def score(batch, model, norm=False):
        rez = model.transform(batch)
        if norm:
            # minmax norm over each query score set 
            rez['score'] = rez.groupby('qid')['score'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return rez
   
    main_lookup = defaultdict(dict)

    for subset in tqdm(split_df(triples, ceil(len(triples) / batch_size)), desc="Total Batched Iter"):
        topics = subset[['qid', 'query']].drop_duplicates()
        for i, model in enumerate(models):
            new = pivot_batch(subset.copy())
            res = score(topics, model, norm=True)
            new['score'] = new.apply(lambda x : res.loc[(res.qid == x['qid']) & (res.docno == x['docno'])]['score'].iloc[0], axis=1)
            main_lookup[i].update(convert_to_dict(new))

        ground_truth = pivot_batch(subset.copy())
        absolute_scores = [1. if i % 2 == 0 else 0. for i in range(len(ground_truth))]
        ground_truth['score'] = absolute_scores

        main_lookup[len(models)+1].update(convert_to_dict(ground_truth))    

        
            

    with open(out_path, 'w') as f:
        json.dump(main_lookup, f)

    return "Done!" 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)