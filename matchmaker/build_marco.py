import pyterrier as pt
pt.init()
import ir_datasets
import pandas as pd
from pyterrier_pisa import PisaIndex
import numpy as np
import fire

def main(dataset : str,
         pisa_dataset : str,
         positives_file : str,
         negatives_file : str,
         num_negative : int = 5,
         positive_cutoff : int = 50,
         arbitrary_positive : int = 0,
         cutoff : int = 1000,
         threads : int = 4):
    
    ds = ir_datasets.load(dataset)
    triples = pd.DataFrame(ds.docpairs_iter()).rename(columns={'query_id':'qid', 'doc_id_a':'docno', 'doc_id_b':'nid'})[['qid', 'docno']]
    triples.drop_duplicates(inplace=True)
    queries = pd.DataFrame(ds.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'})

    pisa_index = PisaIndex.from_dataset(pisa_dataset, 'pisa_porter2')
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
    
    negatives_frame = []
    for row in queries.itertuples():
        record = {}
        record['qid'] = row.qid
        for i, n in enumerate(query_frame[row.qid]):
            record[i] = n
        negatives_frame.append(record)

    negatives_frame = pd.DataFrame.from_records(negatives_frame)
    negatives_frame.to_csv(negatives_file, sep='\t', index=False)

    triples.to_csv(positives_file, sep='\t', index=False, header=False)
    
    return 0

if __name__=='__main__':
    fire.Fire(main)