import pyterrier as pt
if not pt.started(): pt.init()
import os
from fire import Fire 
from pyterrier_pisa import PisaIndex 
import ir_datasets as irds
import pandas as pd
import re

def batch_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def compute_all_bm25(index_path : str, 
                     dataset : str, 
                     output_path : str,
                     threads : int = 1,
                     cutoff : int = 1000,
                     verbose : bool = False):
   
    os.makedirs(output_path, exist_ok=True)
    clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)
    ds = irds.load(dataset)

    index = PisaIndex.from_dataset(index_path, threads=threads)
    model = index.bm25(num_results=cutoff, verbose=verbose) 
    
    queries = pd.DataFrame(ds.queries_iter())

    topics = queries.rename(columns={'query_id': 'qid', 'text' : 'query'})
    topics['query'] = topics['query'].apply(lambda x: clean(x))
    results = model.transform(topics)

    counts = results['qid'].value_counts()
    counts = counts[counts >= cutoff]

    results = results[results['qid'].isin(counts['qid'])]

    results.to_json(os.path.join(output_path, f'bm25.{cutoff}.results.json'), orient='records')

    print('Completed BM25')

        # .agg({'docno': list}).rename(columns={'docno': 'doc_id_b'}).reset_index()
        

if __name__ == '__main__':
    Fire(compute_all_bm25)