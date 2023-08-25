import pandas as pd
import json
import torch
from typing import Any

class TeacherLoader:
    teacher = None 
    triples = None
    tokenizer_kwargs = {'padding' : 'longest', 'truncation' : True, 'return_tensors' : 'pt'}
    def __init__(self, 
                 teacher_file : str, 
                 triples_file : str, 
                 corpus : Any,
                 tokenizer : Any,
                 batch_size : int = 16,
                 shuffle : bool = False,
                 tokenizer_kwargs : dict = None) -> None:
        self.teacher_file = teacher_file
        self.triples_file = triples_file
        self.tokenizer = tokenizer
        self.corpus = corpus

        if tokenizer_kwargs is not None: self.tokenizer_kwargs.update(tokenizer_kwargs)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.initialized = False

    def setup(self) -> None:
        with open(self.teacher_file, 'r') as f:
            self.teacher = json.load(f)
        self.triples = pd.read_csv(self.triples_file, sep='\t', dtype={'qid':str, 'doc_id_a':str, 'doc_id_b':str}, index_col=False)
        if self.shuffle: self.triples = self.triples.sample(frac=1).reset_index(drop=True)
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()

        self.initialized = True

    def get_teacher_scores(self, qid, doc_id, neg=False) -> torch.Tensor:
        sample = []
        for _, teacher in self.teacher.items():
            try:
                score = teacher[str(qid)][str(doc_id)]
            except KeyError:
                score = 0. if neg else 1.
            score = teacher[str(qid)][str(doc_id)]
            sample.append(score)

        return torch.tensor(sample)

    def format(self, q, d):
        return 'Query: ' + q + ' Document: ' + d + ' Relevant:'
    
    def tokenize(self, x):
        return self.tokenizer(x, **self.tokenizer_kwargs)
    
    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        x = [self.format(self.queries[item['qid']], self.docs[item['doc_id_a']]), self.format(self.queries[item['qid']], self.docs[item['doc_id_b']])]
        y = [self.get_teacher_scores(item['qid'], item['doc_id_a'], neg=False), self.get_teacher_scores(item['qid'], item['doc_id_b'], neg=True)]

        return x, y

    def get_batch(self, idx):
        xs = []
        ys = []
        for i in range(idx, min(len(self.triples), idx + self.batch_size)):
            x, y = self[i]
            xs.extend(x)
            ys.extend(y)
        return self.tokenize(xs), torch.cat(ys, dim=0)