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
                     negative_lookup : str = None,
                     threads : int = 1,
                     subsample : int = 100000,
                     cutoff : int = 1000,
                     verbose : bool = False):
   
    os.makedirs(output_path, exist_ok=True)
    clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)
    ds = irds.load(dataset)
    docpairs = pd.DataFrame(ds.docpairs_iter())

    if not negative_lookup:

        index = PisaIndex.from_dataset(index_path, threads=threads)
        model = index.bm25(num_results=cutoff, verbose=verbose) 
        
        queries = pd.DataFrame(ds.queries_iter())

        topics = queries.rename(columns={'query_id': 'qid', 'text' : 'query'})
        topics['query'] = topics['query'].apply(lambda x: clean(x))
        results = model.transform(topics)

        print('Completed BM25')

        # .agg({'docno': list}).rename(columns={'docno': 'doc_id_b'}).reset_index()

        BATCH_SIZE = 10000
        qid_groups = list(results[['qid', 'docno']].groupby('qid'))

        tmp = []

        for batch in batch_iter(qid_groups, BATCH_SIZE):
            print(f'Processing batch of {len(batch)}')
            batch = pd.DataFrame(batch, columns=['qid', 'docno'])
            batch = batch.groupby('qid').agg({'docno': list}).rename(columns={'docno': 'doc_id_b'}).reset_index()
            batch['length'] = batch['doc_id_b'].apply(lambda x: len(x))
            batch = batch[batch['length'] >= cutoff]
            batch['doc_id_b'] = batch['doc_id_b'].apply(lambda x: x[::-1])
            tmp.append(batch[['qid', 'doc_id_b']])
        
        results = pd.concat(tmp)

        print('Aggregated')
        
        print(f'{len(results)} valid topics found')
        results.to_json(os.path.join(output_path, f'bm25.{cutoff}.negatives.json'), orient='records')
    else:
        results = pd.read_json(negative_lookup, orient='records')

    negative_lookup = results.set_index('qid')['doc_id_b'].to_dict()
    del results

    all_topic_ids = docpairs['query_id'].unique().tolist()
    all_negative_ids = results['qid'].unique().tolist()

    candidates = list(set(all_topic_ids).intersection(set(all_negative_ids)))

    docpairs = docpairs[docpairs['query_id'].isin(candidates)].copy()
    docpairs = docpairs.sample(subsample)[['query_id', 'doc_id_a']]
    docpairs['doc_id_b'] = docpairs['query_id'].apply(lambda x: negative_lookup[x])
    
    docpairs.to_json(os.path.join(output_path, f'bm25.{cutoff}.{subsample}.json'), orient='records')

if __name__ == '__main__':
    Fire(compute_all_bm25)