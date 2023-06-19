import torch.utils.data.Dataset as Dataset
import pandas as pd
from math import floor, ceil

class TripletDataset(Dataset):
    def __init__(self, pairs, neg_idx, corpus, max=False):
        self.data = pairs
        self.neg_idx = neg_idx
        self.n_neg = neg_idx.shape[-1]
        self.docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        self.round = ceil if max else floor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx, weights=None):
        assert weights, "Weights not set"
        q, p = self.data[idx]
        n = self.neg_idx[idx][self.round(self.weights[idx].item() * self.n_neg)]
        return q, p, self.docs[n]

class TripletLoader:
    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        self.num_items = len(dataset)
        self.batch_size = batch_size
    
    def get_batch(self, idx, weights):
        q, p, n = [], [], []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            _q, _p, _n = self.dataset[i, weights]
            q.append(_q)
            p.append(_p)
            n.append(_n)
        return q, p, n

        
