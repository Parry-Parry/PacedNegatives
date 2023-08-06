import pyterrier as pt
if not pt.started(): pt.init()
import os
from fire import Fire 
from pyterrier_pisa import PisaIndex 
import ir_datasets as irds
import pandas as pd
import re

def compute_all_bm25(index_path : str, 
                     dataset : str, 
                     output_path : str,
                     threads : int = 1,
                     subsample : int = 100000,
                     cutoff : int = 1000,
                     verbose : bool = False):
   
    os.makedirs(output_path, exist_ok=True)
    clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

    index = PisaIndex.from_dataset(index_path, threads=threads)
    model = index.bm25(num_results=cutoff, verbose=verbose) 

    #model = pt.BatchRetrieve.from_dataset(index_path, 'terrier_stemmed', wmodel="BM25", verbose=verbose)

    ds = irds.load(dataset)
    docpairs = pd.DataFrame(ds.docpairs_iter()).drop_duplicates(['query_id'])
    positive_lookup = docpairs.set_index('query_id')['doc_id_a'].to_dict()
    queries = pd.DataFrame(ds.queries_iter())

    topics = queries.rename(columns={'query_id': 'qid', 'text' : 'query'})
    topics['query'] = topics['query'].apply(lambda x: clean(x))

    print(f'searching over {len(topics)} topics')

    results = model.transform(topics)

    results = results[['qid', 'docno']].groupby('qid').agg({'docno': list}).rename(columns={'docno': 'doc_id_b'}).reset_index()
    results = results[len(results['doc_id_b'] >= cutoff)]
    subresults = results.sample(subsample)
    subresults['doc_id_b'] = subresults['doc_id_b'].apply(lambda x: x[::-1])
    subresults['doc_id_a'] = subresults['qid'].apply(lambda x: positive_lookup[x])
    
    subresults.to_json(os.path.join(output_path, f'bm25.{cutoff}.{subsample}.json'), orient='records')

if __name__ == '__main__':
    Fire(compute_all_bm25)