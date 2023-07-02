import pandas as pd
import numpy as np
import re

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return re.sub(r'/[^\x00-\x7F]/g', '', text).strip()

class adhocRestructure:
    def __init__(self, model, corpus) -> None:
        self.model = model
        self.queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
        self.docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
    
    def __call__(self, q, idx):
        tmp_frame = pd.DataFrame({'qid': [q] * len(idx), 'query' : [clean_text(self.queries[q])] * len(idx), 'docno': idx, 'text' : [clean_text(self.docs[i]) for i in idx]})
        scored = self.model.transform(tmp_frame)
        return scored.sort_values('score', ascending=False).docno.tolist()

def collapse_triples(triples, model, corpus, num_docs=0):
    """
    Collapses triples dataframe by grouping by q_id and doc_id_a such that the form becomes a single query, a single doc_id_a and a list of doc_id_bs
    """
    model = adhocRestructure(model, corpus)
    new_df = triples.groupby(['query_id', 'doc_id_a']).agg({'doc_id_b': list}).reset_index()
    new_df['doc_id_b'] = new_df.apply(lambda x : model(x['query_id'], x['doc_id_b']), axis=1)
    if num_docs: new_df['doc_id_b'] = new_df['doc_id_b'].apply(lambda x : x[:num_docs])
    return new_df[['query_id', 'doc_id_a', 'doc_id_b']]

def take_subset(triples, num_docs=10):
    triples = triples.copy()
    triples['doc_id_b'] = triples['doc_id_b'].apply(lambda x : x[:num_docs])
    return triples

def get_balanced_idx(vals, num_docs):
    spacing = np.linspace(0, len(vals)-1, num_docs, endpoint=True, dtype=int)
    return [vals[i] for i in spacing]

def take_balanced_subset(triples, num_docs=10):
    triples = triples.copy()
    triples['doc_id_b'] = triples['doc_id_b'].apply(lambda x : get_balanced_idx(x, num_docs))
    return triples