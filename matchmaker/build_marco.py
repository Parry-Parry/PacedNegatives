import pyterrier as pt
pt.init()
import ir_datasets
import pandas as pd
from pyterrier_pisa import PisaIndex
import numpy as np
import fire

def main(dataset :str,
         out_file : str,
         num_negative : int = 5,
         positive_cutoff : int = 50,
         arbitrary_positive : int = 0,
         cutoff : int = 1000,
         threads : int = 4):
    
    ds = ir_datasets.load(dataset)
    triples = pd.DataFrame(ds.docpairs_iter()).rename(columns={'query_id':'qid', 'doc_id_a':'docno', 'doc_id_b':'nid'})[['qid', 'docno']]
    triples.drop_duplicates(inplace=True)
    queries = pd.DataFrame(ds.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'})

    pisa_index = PisaIndex.from_dataset('msmarco_passage', 'pisa_porter2')
    index = pisa_index.bm25(num_results=cutoff, threads=threads)

    scores = index.transform(queries)
    spacing = np.linspace(0, cutoff-positive_cutoff , num=num_negative).astype(np.int8)
    
    query_frame = {}
    for row in queries.itertuples():
        subset = scores[scores.qid == row.qid].sort_values(by='score', ascending=False)
        negative = subset.iloc[positive_cutoff:cutoff]
        query_frame[row.qid] = negative.iloc[spacing].tolist()
        if arbitrary_positive:
            positive = subset.iloc[:positive_cutoff]
            samples = positive.sample(arbitrary_positive)[['qid', 'docno']]
            triples.append(samples)
    
    new_frame = []
    for row in triples.itertuples():
        record = {}
        record['qid'] = row.qid
        record['pid'] = row.docno
        for i, n in enumerate(query_frame[row.qid]):
            record[i] = n
        new_frame.append(record)

    new_frame = pd.DataFrame.from_records(new_frame)
    new_frame.to_csv(out_file, sep='\t', index=False)
    
    return 0

if __name__=='__main__':
    fire.Fire(main)