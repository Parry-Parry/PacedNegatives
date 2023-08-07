import pyterrier as pt
if not pt.started(): pt.init()
import os
from fire import Fire 
from pyterrier_pisa import PisaIndex 
from pyt_splade import SpladeFactory
import ir_datasets as irds
import pandas as pd
import re

def batch_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable.iloc[ndx:min(ndx + n, l)]

def compute_all_splade(index_path : str, 
                     model_name_or_path : str,
                     dataset : str, 
                     output_path : str,
                     threads : int = 1,
                     cutoff : int = 1000,
                     verbose : bool = False):
   
    os.makedirs(output_path, exist_ok=True)
    clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)
    ds = irds.load(dataset)

    index = PisaIndex.from_dataset(index_path, threads=threads)
    splade = SpladeFactory(model_name_or_path)
    model = splade.query_encoder(scale=100.) >> index.quantized(num_results=cutoff, verbose=verbose) % cutoff  

    queries = pd.DataFrame(ds.queries_iter())

    topics = queries.rename(columns={'query_id': 'qid', 'text' : 'query'})
    topics['query'] = topics['query'].apply(lambda x: clean(x))

    tmp = []
    for batch in batch_iter(topics, 1000):
        results = model.transform(batch)
        counts = results['qid'].value_counts()
        counts = counts[counts >= cutoff]
        candidates = counts.index.tolist()
        results = results[results['qid'].isin(candidates)]

        results = results.groupby('qid').agg({'docno': list}).rename(columns={'docno': 'doc_id_b'}).reset_index()
        results['doc_id_b'] = results['doc_id_b'].apply(lambda x: x[:cutoff])
        results['doc_id_b'] = results['doc_id_b'].apply(lambda x: x[::-1])

        tmp.append(results[['qid', 'doc_id_b']].copy())

    results = pd.concat(tmp)
    results.to_json(os.path.join(output_path, f'bm25.{cutoff}.results.json'), orient='records')

    print('Completed BM25')

if __name__ == '__main__':
    Fire(compute_all_splade)


    

