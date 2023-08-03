import pyterrier as pt
if not pt.started(): pt.init()
import os
from fire import Fire 
from pyterrier_pisa import PisaIndex 
from pyt_splade import SpladeFactory
import ir_datasets as irds
import pandas as pd

def compute_all_bm25(index_path : str, 
                     model_name_or_path : str,
                     dataset : str, 
                     output_path : str,
                     threads : int = 1,
                     subsample : int = 100000,
                     cutoff : int = 1000):
   
    os.makedirs(output_path, exist_ok=True)

    index = PisaIndex.from_dataset(index_path, threads=threads)
    splade = SpladeFactory(model_name_or_path)
    model = splade.query_encoder(scale=100.) >> index.quantized(num_results=cutoff) % cutoff  

    ds = irds.load(dataset)
    docpairs = pd.DataFrame(ds.docpairs_iter()).drop_duplicates('query_id').sample(subsample)
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()

    docpairs['query'] = docpairs['query_id'].apply(lambda x: queries[x])

    topics = docpairs[['query_id', 'query']].rename(columns={'query_id': 'qid'})
    results = model.transform(topics)

    results = results.groupby('qid').agg({'docno': list}).reset_index()
    results.to_json(os.path.join(output_path, f'splade.{cutoff}.{subsample}.json'), orient='records')

if __name__ == '__main__':
    Fire(compute_all_bm25)


    

