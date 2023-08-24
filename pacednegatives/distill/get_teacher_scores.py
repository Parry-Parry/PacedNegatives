from collections import defaultdict
import pyterrier as pt 
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
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
    
    triples = pd.read_csv(triples_path, sep="\t", dtype={"qid":str, "doc_id_a":str, "query":str, "doc_id_b":str}, index_col=False)
    dataset = irds.load(corpus_lookup)
    docs = pd.DataFrame(dataset.docs_iter()).set_index("doc_id")["text"].to_dict()

    index = pt.get_dataset(index_path).get_index('terrier_stemmed')
    index = pt.IndexFactory.of(index, memory=True)

    #properties = { 'querying.processes' : pt.BatchRetrieve.default_properties['querying.processes'].replace('qe:QueryExpansion', 'qe:QueryExpansion,rm1:RM1,rm3:RM3') }
    properties = {
        'querying.processes' : "terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,localmatching:LocalManager$ApplyLocalMatching,rm1:RM1,rm3:RM3,ax:AxiomaticQE,qe:QueryExpansion,labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess'"
    }
    models = [
        pt.text.scorer(body_attr="text", wmodel="BM25", controls={"qe":"on", "qemodel" : "Bo1"}, background_index=index),
        pt.text.scorer(body_attr="text", wmodel="BM25", controls={"qe":"on", "qemodel" : "KL"}, background_index=index),
        pt.text.scorer(body_attr="text", wmodel="BM25", controls={"qe":"on", "rm3" : "on"}, background_index=index, properties=properties),
        pt.text.scorer(body_attr="text", wmodel="DPH", controls={"qe":"on", "qemodel" : "Bo1"}, background_index=index),
        pt.text.scorer(body_attr="text", wmodel="DPH", controls={"qe":"on", "qemodel" : "KL"}, background_index=index),
    ]
    
    def pivot_batch(batch):
        print(batch.columns)
        print(batch.head(5))
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
        lookup = defaultdict(lambda : defaultdict(lambda : 0))
        for row in result.itertuples():
            lookup[row.qid][row.docno] = row.score
        return lookup
    
    def score(batch, model, norm=False):
        rez = model.transform(batch)
        if norm:
            # group by query and minmax normalise score 
            rez['score'] = rez.groupby('qid')['score'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return convert_to_dict(rez)
        
    main_lookup = defaultdict(dict)

    for batch in tqdm(split_df(triples, ceil(len(triples) / batch_size))):
        to_score = pivot_batch(batch)
        for i, model in enumerate(models):
            main_lookup[i].update(score(to_score, model, True))
        
        # add score to each row which alternates as 1 and 0
        absolute_scores = [1. if i % 2 == 0 else 0. for i in range(len(to_score))]
        to_score['score'] = absolute_scores

        main_lookup[len(models)+1].update(convert_to_dict(to_score))
        
    
    with open(out_path, 'w') as f:
        json.dump(main_lookup, f)

    return "Done!" 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)

