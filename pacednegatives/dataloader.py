from torch.autograd import Variable
import torch
import pandas as pd
from math import floor, ceil

gen_var = lambda x, y : Variable(x, requires_grad=y)

OUTPUTS = ["TRUE", "FALSE"]

class TripletDataset:
    def __init__(self, pairs, neg_idx, corpus, max=False):
        self.neg_idx = neg_idx
        self.n_neg = len(neg_idx[0])
        self.docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        self.queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
        self.round = ceil if max else floor

        self.data = [(self.queries[q], self.docs[p]) for q, p in pairs]

    def __len__(self):
        return len(self.data)
    
    def get_items(self, idx, weights=None):
        assert weights is not None, "Weights not set"
        q, p = self.data[idx]
        n = self.neg_idx[idx][self.round(weights[idx].item() * self.n_neg)]
        return self.queries[q], self.docs[p], self.docs[n]

class TripletLoader:
    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        self.num_items = len(dataset)
        self.batch_size = batch_size
    
    def __len__(self):
        return self.num_items
    
    def get_batch(self, idx, weights=None):
        q, p, n = [], [], []
        if weights is None: weights = gen_var(torch.ones(self.batch_size), True).to(self.device)
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            _q, _p, _n = self.dataset.get_items(i, weights)
            q.append(_q)
            p.append(_p)
            n.append(_n)
        return q, p, n

class PairLoader:
    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.dataset)
    
    def format(self, q, d):
        return 'Query: ' + q + ' Document: ' + d + ' Relevant:'
    
    def get_batch(self, idx, weights=None):
        px, nx, o_p, o_n = [], [], [], []
        if weights is None: weights = gen_var(torch.ones(self.batch_size), True).to(self.device)
        for j in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            q, p, n = self.dataset.get_items(j, weights)
            for _q, _p, _n in zip(q, p, n):
                px.append(self.format(_q, _p))
                nx.append(self.format(_q, _n))
                o_p.append(OUTPUTS[0])
                o_n.append(OUTPUTS[1])
        return px, nx, o_p, o_n

        
