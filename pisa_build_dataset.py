from collections import defaultdict
import pyterrier as pt 
pt.init()
from pyterrier_pisa import PisaIndex
from fire import Fire
from paced.util import *
import re
import ir_datasets
import pandas as pd

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return re.sub(r'/[^\x00-\x7F]/g', '', text).strip()

def make_score(df):
    tmp = defaultdict(dict)
    for row in df.itertuples():
        tmp[row.qid][row.docno] = row.score
    return tmp

def collapse_triples(triples, model, corpus):
    """
    Collapses triples dataframe by grouping by q_id and doc_id_a such that the form becomes a single query, a single doc_id_a and a list of doc_id_bs
    """
    #model = adhocRestructure(model, corpus)

    queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
    tmp_queries = pd.DataFrame({'qid': triples['query_id'].unique(), 'query' : [clean_text(queries[q]) for q in triples['query_id'].unique()]})
    scores = model.transform(tmp_queries)

    score_query = make_score(scores)

    def get_ordered(qid, idx):
        _scores = score_query[qid]
        def scoring(docno):
            try:
                return _scores[docno]
            except KeyError:
                return 0
        
        result = sorted(idx, key=lambda x : scoring(x), reverse=True)
        counts = len([i for i in result if i not in _scores])
        print('QID {qid} Failed to find {counts} docs'.format(qid=qid, counts=counts))
        return result
    
    new_df = triples.groupby(['query_id', 'doc_id_a']).agg({'doc_id_b': list}).reset_index()
    new_df['doc_id_b'] = new_df.apply(lambda x : get_ordered(x['query_id'], x['doc_id_b']), axis=1)
    return new_df[['query_id', 'doc_id_a', 'doc_id_b']]

def main(dataset : str, out_dir : str, res=1500):
    pisa_index = PisaIndex.from_dataset('msmarco_passage', 'pisa_porter2')
    index = pisa_index.bm25(num_results=res, threads=8)
    
    dataset = ir_datasets.load(dataset)
    triples = pd.DataFrame(dataset.docpairs_iter())

    new_df = collapse_triples(triples, index, dataset)
    new_df.to_json(out_dir, orient='records', lines=True)

if __name__ == '__main__':
    Fire(main)