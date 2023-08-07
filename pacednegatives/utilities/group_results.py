import pyterrier as pt
if not pt.started(): pt.init()
import os
from fire import Fire 
import ir_datasets as irds
import pandas as pd

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
    negative_lookup = results.set_index('qid')['doc_id_b'].to_dict()
    all_negative_ids = [str(x) for x in results['qid'].unique().tolist()]
    del results

    print(f'loaded results with {len(all_negative_ids)} qids')

    print(all_negative_ids[:10])
    print(docpairs['query_id'].unique().tolist()[:10])

    docpairs = docpairs[docpairs['query_id'].isin(all_negative_ids)].copy()
    docpairs = docpairs.sample(subsample)[['query_id', 'doc_id_a']]
    docpairs['doc_id_b'] = docpairs['query_id'].apply(lambda x: negative_lookup[int(x)])
    
    docpairs.to_json(os.path.join(output_path, f'{model}.{cutoff}.{subsample}.json'), orient='records')

    
if __name__ == '__main__':
    Fire(compute_all)




       