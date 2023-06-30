import pandas as pd

class adhocRestructure:
    def __init__(self, model, corpus) -> None:
        self.model = model
        self.queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
        self.docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
    
    def __call__(self, q, idx):
        tmp_frame = pd.DataFrame({'query_id': [q] * len(idx), 'query' : [self.queries[q]] * len(idx), 'doc_id': idx, 'text' : [self.docs[i] for i in idx]})
        scored = self.model.score(tmp_frame)
        return scored.sort_values('score', ascending=False).doc_id.tolist()

def collapse_triples(triples, model, corpus, num_docs=0):
    """
    Collapses triples dataframe by grouping by q_id and doc_id_a such that the form becomes a single query, a single doc_id_a and a list of doc_id_bs
    """
    model = adhocRestructure(model, corpus)
    new_df = triples.groupby(['q_id', 'doc_id_a']).agg({'doc_id_b': list}).reset_index()
    new_df['doc_id_b'] = new_df.apply(lambda x : model(x['qid'], x['doc_id_b']), axis=1)
    if num_docs: new_df['doc_id_b'] = new_df['doc_id_b'].apply(lambda x : x[:num_docs])
    return new_df[['q_id', 'doc_id_a', 'doc_id_b']]

