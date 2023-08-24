from collections import defaultdict
import pyterrier as pt 
if not pt.started():
    pt.init()
from pyterrier.model import split_df
from fire import Fire
import pandas as pd
import logging
import ir_datasets as irds
from math import ceil
from tqdm import tqdm
import json

def main(triples_path : str,
         index_path : str,
         corpus_lookup : str,
         out_path : str,
         batch_size : int = 1000,) -> str:
    
    triples = pd.read_csv(triples_path, sep="\t", names=["qid", "doc_id_a", "query" "doc_id_b"])
    dataset = irds.load(corpus_lookup)
    docs = pd.DataFrame(dataset.docs_iter()).set_index("doc_id")["text"].to_dict()

    index = pt.get_dataset(index_path).get_index('terrier_stemmed')
    index = pt.IndexFactory.of(index, memory=True)

    properties = { 'querying.processes' : pt.BatchRetrieve.default_properties['querying.processes'].replace('qe:QueryExpansion', 'qe:QueryExpansion,rm1:RM1,rm3:RM3') }

    models = [
        pt.text.TextScorer(body_attr="text", wmodel="BM25", controls={"qe":"on", "qemodel" : "Bo1"}, background_index=index),
        pt.text.TextScorer(body_attr="text", wmodel="BM25", controls={"qe":"on", "qemodel" : "KL"}, background_index=index),
        pt.text.TextScorer(body_attr="text", wmodel="BM25", controls={"qe":"on", "qemodel" : "rm3"}, background_index=index, properties=properties),
        pt.text.TextScorer(body_attr="text", wmodel="DPH", controls={"qe":"on", "qemodel" : "Bo1"}, background_index=index),
        pt.text.TextScorer(body_attr="text", wmodel="DPH", controls={"qe":"on", "qemodel" : "KL"}, background_index=index),
    ]
    
    def pivot_batch(batch):
        records = []
        for row in batch.itertuples():
            records.extend([{
                'qid': row.qid,
                'query': row.query,
                'docno': row.doc_id_a,
                'text': docs[row.doc_id_a]
                },
                {
                'qid': row.qid,
                'query': row.query,
                'docno': row.doc_id_b,
                'text': docs[row.doc_id_b],
                }
                ])
        return pd.DataFrame.from_records(records)

    def convert_to_dict(result):
        lookup = defaultdict(defaultdict(lambda : 0))
        for row in result.itertuples():
            lookup[row.qid][row.docno] = row.score
        return lookup
    
    def score(batch, model, norm=False):
        rez = model.score(batch)
        if norm:
            # group by query and minmax normalise score 
            rez['score'] = rez.groupby('qid')['score'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return convert_to_dict(rez)
        
    main_lookup = defaultdict(defaultdict(lambda : 0))

    for batch in tqdm(split_df(triples, ceil(len(triples) / batch_size))):
        to_score = pivot_batch(batch)
        for i, model in enumerate(models):
            main_lookup[i].update(score(to_score, model, True))
        
        # add score to each row which alternates as 1 and 0
        absolute_scores = [1. if i % 2 == 0 else 0. for i in range(len(to_score))]
        to_score['score'] = absolute_scores

        main_lookup[len(models)+1].append(convert_to_dict(to_score))
        
    
    with open(out_path, 'w') as f:
        json.dump(main_lookup, f)

    return "Done!" 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)

