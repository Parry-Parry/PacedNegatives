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

def compute_all(model : str,
                dataset : str, 
                output_path : str,
                results_path : str,
                subsample : int = 100000,
                cutoff : int = 1000,
                ):
   
    os.makedirs(output_path, exist_ok=True)
    ds = irds.load(dataset)
    docpairs = pd.DataFrame(ds.docpairs_iter())
    results = pd.read_json(results_path, orient='records')
    results = results[['qid', 'docno']]

    # count the number of docnos for each qid without groupby
    results['count'] = results['qid'].map(results['qid'].value_counts())

    results = results[results['count'] >= cutoff]

    print('Filtered')

    results = results.groupby('qid').agg({'docno': list}).rename(columns={'docno': 'doc_id_b'}).reset_index()
    results['doc_id_b'] = results['doc_id_b'].apply(lambda x: x[::-1])

    '''

    BATCH_SIZE = 10000
    qid_groups = batch_iter(results.groupby('qid'), BATCH_SIZE)

    tmp = []

    for batch in qid_groups:
        print(f'Processing batch of {len(batch)}')
        batch = pd.DataFrame(batch, columns=['qid', 'docno'])
        batch = batch.groupby('qid').agg({'docno': list}).rename(columns={'docno': 'doc_id_b'}).reset_index()
        batch['length'] = batch['doc_id_b'].apply(lambda x: len(x))
        batch = batch[batch['length'] >= cutoff]
        batch['doc_id_b'] = batch['doc_id_b'].apply(lambda x: x[::-1])
        tmp.append(batch[['qid', 'doc_id_b']])
    
    results = pd.concat(tmp)
    '''
    print('Aggregated')

    negative_lookup = results.set_index('qid')['doc_id_b'].to_dict()
    del results

    all_topic_ids = docpairs['query_id'].unique().tolist()
    all_negative_ids = results['qid'].unique().tolist()

    candidates = list(set(all_topic_ids).intersection(set(all_negative_ids)))

    docpairs = docpairs[docpairs['query_id'].isin(candidates)].copy()
    docpairs = docpairs.sample(subsample)[['query_id', 'doc_id_a']]
    docpairs['doc_id_b'] = docpairs['query_id'].apply(lambda x: negative_lookup[x])
    
    docpairs.to_json(os.path.join(output_path, f'{model}.{cutoff}.{subsample}.json'), orient='records')

    
if __name__ == '__main__':
    Fire(compute_all)




       