from functools import partial
from typing import Any

import numpy as np
from torch.autograd import Variable
import torch
import pandas as pd
from scipy.stats import binom
from math import floor, ceil
from torch.utils.data import Dataset

gen_var = lambda x, y : Variable(x, requires_grad=y)

OUTPUTS = ["true", "false"]

class TripletDataset:
    def __init__(self, pairs, neg_idx, corpus, max=False):
        self.neg_idx = neg_idx
        self.n_neg = len(neg_idx[0]) - 1
        self.docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        self.queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
        self.round = ceil if max else floor

        self.data = [(self.queries[q], self.docs[p]) for q, p in pairs]

    def __len__(self):
        return len(self.data)
    
    def get_items(self, idx, weight=None):
        assert weight is not None, "Weights not set"
        q, p = self.data[idx]
        n = self.neg_idx[idx][self.round(weight * self.n_neg)]
        return q, p, self.docs[n]

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
        for i, j in enumerate(range(idx * self.batch_size, (idx + 1) * self.batch_size)):
            q, p, n = self.dataset.get_items(j, weights[i].item())
            px.append(self.format(q, p))
            nx.append(self.format(q, n))
            o_p.append(OUTPUTS[0])
            o_n.append(OUTPUTS[1])

        return px, nx, o_p, o_n

class LevelLoader:
    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.dataset)
    
    def format(self, q, d):
        return 'Query: ' + q + ' Document: ' + d + ' Relevant:'
    
    def get_batch(self, idx, weight=None):
        px, nx, o_p, o_n = [], [], [], []
        if weight is None: weight = 0.0
        for j in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            q, p, n = self.dataset.get_items(j, weight)
            px.append(self.format(q, p))
            nx.append(self.format(q, n))
            o_p.append(OUTPUTS[0])
            o_n.append(OUTPUTS[1])

        return px, nx, o_p, o_n
'''
class LCEDataset:
    def __init__(self, pairs, neg_idx, corpus,max=False):
        self.neg_idx = [np.array(n) for n in neg_idx]
        self.n_neg = len(neg_idx[0]) 
        self.docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        self.queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
        self.round = ceil if max else floor

        self.data = [(self.queries[q], self.docs[p]) for q, p in pairs]

    def __len__(self):
        return len(self.data)
    
    def get(self, idx):
        return self.data[idx[0]], [self.docs[x] for x in self.neg_idx[idx[0]][idx[1]].tolist()]
  '''  
class LCEDataset(Dataset):
    def __init__(self, 
                 pairs, 
                 neg_idx, 
                 corpus, 
                 tokenizer,
                 init_weight,
                 batch_size : int, 
                 var : float, 
                 n : int, 
                 min : float, 
                 max : float, 
                 use_max=False,) -> None:
        
        self.neg_idx = [np.array(n) for n in neg_idx]
        self.n_neg = len(neg_idx[0]) 
        self.docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        self.queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
        self.round = ceil if use_max else floor
        self.tokenizer = tokenizer
        self.data = [(self.queries[q], self.docs[p]) for q, p in pairs]

        self.batch_size = batch_size
        self.var = var
        self.n = n
        self.round = torch.floor
        self.min = min
        self.max = max

        self.weight = init_weight

    def get(self, idx):
        return self.data[idx[0]], [self.docs[x] for x in self.neg_idx[idx[0]][idx[1]].tolist()]
    
    def __len__(self):
        return len(self.data)
    
    def sample(self, mean):
        mean = np.clip(mean, self.min, self.max)
        n = self.n_neg - 1
        idx = np.arange(self.n_neg)

        probabilities = binom.pmf(idx, n, mean)
        probabilities = binom.pmf(idx, n, mean)
        adjusted_probabilities = probabilities / probabilities.sum()
        adjusted_variance = np.var(adjusted_probabilities)
        scaling_factor = np.sqrt(self.var / adjusted_variance)
        adjusted_probabilities *= scaling_factor
        adjusted_probabilities /= adjusted_probabilities.sum()
        try:
            samples =  np.random.choice(idx, size=(self.n,), replace=False, p=adjusted_probabilities)
            return samples
        except ValueError:
            print(f'Failed to sample with mean {mean} and var {self.var}')
            exit()
    
    def format(self, q, d):
        x = 'Query: ' + q + ' Document: ' + d + ' Relevant:'
        return x

    def __getitem__(self, idx):
        _idx = self.sample(self.weight)
        qp, n = self.get((idx, _idx))
        q, p = qp

        return {'pos' : self.format(q, p), 'neg' : list(map(partial(self.format, q), n))}
     
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
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size-1):
            _q, _p, _n = self.dataset.get_items(i, weights[i].item())
            print(_q)
            q.append(_q)
            p.append(_p)
            n.append(_n)
        return q, p, n


        
