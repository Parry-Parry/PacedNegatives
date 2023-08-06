import pyterrier as pt
if not pt.started(): pt.init()
import os
from fire import Fire 
from pyterrier_pisa import PisaIndex 
import ir_datasets as irds
import pandas as pd

def compute_all_bm25(index_path : str, 
                     dataset : str, 
                     output_path : str,
                     threads : int = 1,
                     subsample : int = 100000,
                     cutoff : int = 1000,
                     verbose : bool = False):
   
    os.makedirs(output_path, exist_ok=True)

    index = PisaIndex.from_dataset(index_path, threads=threads)
    model = index.bm25(num_results=cutoff, verbose=verbose) 

    ds = irds.load(dataset)
    docpairs = pd.DataFrame(ds.docpairs_iter()).sample(subsample)[['query_id', 'doc_id_a']]
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()

    all_possible = docpairs.drop_duplicates('query_id').copy()
    all_possible['query'] = all_possible['query_id'].apply(lambda x: queries[x])

    topics = all_possible[['query_id', 'query']].rename(columns={'query_id': 'qid'})
    results = model.transform(topics)

    results = results.groupby('qid').agg({'docno': list}).rename(columns={'docno': 'doc_id_b'}).reset_index()
    results['doc_id_b'] = results['doc_id_b'].apply(lambda x: x[:cutoff][::-1])
    negative_dict = results.set_index('qid')['doc_id_b'].to_dict()
    
    docpairs['doc_id_a'] = docpairs['query_id'].apply(lambda x: negative_dict[x])

    results.to_json(os.path.join(output_path, f'bm25.{cutoff}.{subsample}.json'), orient='records')

if __name__ == '__main__':
    Fire(compute_all_bm25)