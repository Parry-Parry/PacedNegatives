from functools import partial
from typing import Any

import numpy as np
from torch.autograd import Variable
import torch
import pandas as pd
from scipy.stats import binom
from math import floor, ceil

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
    
class LCELoader:
    def __init__(self, dataset : Any, batch_size : int, var : float, n : int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.var = var
        self.n = n
        self.round = torch.floor
    
    def sample(self, mean):
        n = self.dataset.n_neg - 1
        idx = np.arange(self.dataset.n_neg, dtype=np.int32)

        probabilities = binom.pmf(idx, n, mean)
        adjusted_probabilities = probabilities / probabilities.sum()
        adjusted_variance = np.var(adjusted_probabilities)
        scaling_factor = np.sqrt(self.var / adjusted_variance)
        adjusted_probabilities *= scaling_factor
        adjusted_probabilities /= adjusted_probabilities.sum()

        print(n, self.n, len(np.where(adjusted_probabilities > 0.)))

        return np.random.choice(idx, size=(self.n,), replace=False, p=adjusted_probabilities)

    def torch_sample(self, mean):
        sample2idx = lambda x : self.round(torch.clamp(x, 0.0, 1.0) * self.n_neg)
        initial = torch.unique(sample2idx(torch.normal(mean, self.var, size=(self.n,))), dim=0)
        while len(initial) < self.n:
            new = torch.unique(sample2idx(torch.normal(mean, self.var, size=(self.n - len(initial),))), dim=0)
            initial = torch.unique(torch.cat((initial, new), dim=0))
        return initial.to_numpy()

    def __len__(self):
        return len(self.dataset)
    
    def format(self, q, d):
        return 'Query: ' + q + ' Document: ' + d + ' Relevant:'

    def get_batch(self, idx, weight=None):
        px, nx = [], []
        for j in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            _idx = self.sample(weight)
            qp, n = self.dataset.get((j, _idx))
            q, p = qp
            px.append(self.format(q, p))
            nx.extend(map(partial(self.format, q), n))

        return px, nx
    
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

        
