from collections import defaultdict
import json
import time

import numpy as np
import fire 
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import os 
from math import ceil
import ir_datasets
import pickle
import pandas as pd
import logging

from torch.autograd import Variable, grad

gen_param = lambda x, y : nn.Parameter(torch.Tensor([x]), requires_grad=y)
gen_var = lambda x, y : Variable(x, requires_grad=y)

RND = 42
OUTPUTS = ['true', 'false']

loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')

def process_dataset(dataset, cut=None):
    frame = pd.DataFrame(dataset.docpairs_iter())
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id').text.to_dict()

    frame['query'] = frame['query_id'].apply(lambda x: queries[x])
    frame['pid'] = frame['doc_id_a'].apply(lambda x: docs[x])
    frame['nid'] = frame['doc_id_b'].apply(lambda x: docs[x])
    if cut: frame = frame.sample(cut, random_state=RND) 
    return frame[['query', 'pid', 'nid']]

def load_t5(model_name : str = 't5-base'):
    return T5ForConditionalGeneration.from_pretrained(model_name)

def set_param(curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

def update_params(model, lr, grads):
    for tgt, src in zip(model.named_parameters(), grads):
        name_t, param_t = tgt
        grad = src
        tmp = nn.Parameter(param_t - lr * grad, requires_grad=True)
        set_param(model, name_t, tmp)

class Weights(nn.Module):
    def __init__(self, eta : float, device = None, min=np.log(2), max=10):
        super().__init__()
        self.clamp = lambda x : torch.clamp(x, min=min, max=max)
        self.eta = self.clamp(nn.parameter(torch.tensor([eta]), requires_grad=True)).to(device)

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, loss, eta=None):
        weight = gen_var(torch.zeros(loss.size()), True)

        for i in range(len(loss)):
            if loss[i] > self.eta:
                weight[i] = torch.zeros(1).to(self.device) if eta else torch.zeros(1).to(self.device)*self.eta
            else:
                weight[i] = (-loss[i] / eta) + 1 if eta else (-loss[i] / self.eta) + 1
        return weight
        

torch.manual_seed(RND)
_logger = ir_datasets.log.easy()

def main(dataset : str, 
         out_dir : str,
         model_name : str = 't5-base',
         epochs : int = 10, 
         batch_size : int = 128,
         lr : float = 5e-5,
         meta_lr : float = None,
         cut : int = None,
         eta : float = 5.0,
         max_eta : float = 10.0,):
    
    logs = {
        'dataset': dataset,
        'model_name': model_name,
        'spl': True,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'meta_lr': meta_lr if meta_lr else lr,
        'cut': cut if cut else -1,
        'eta': defaultdict(list),
        'loss' : defaultdict(list),
        'zeros' : defaultdict(list),
    }

    os.makedirs(out_dir, exist_ok=True)
    df  = process_dataset(ir_datasets.load(dataset), cut=cut)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_t5(model_name).to(device)
    meta_model = load_t5(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    optimizer = AdamW(model.parameters(), lr=lr)

    if not meta_lr: meta_lr = lr
    weights = Weights(eta, device=device, max=max_eta)

    def iter_train_samples():    
            while True:
                for row in df.itertuples():
                    yield 'Query: ' + row.query + ' Document: ' + row.pid + ' Relevant:', OUTPUTS[0]
                    yield 'Query: ' + row.query + ' Document: ' + row.nid + ' Relevant:', OUTPUTS[1]

    start = time.time()
    train_iter = _logger.pbar(iter_train_samples(), desc='total train samples')

    for epoch in range(epochs):
        with _logger.pbar_raw(desc=f'train {epoch}', total= len(df) // batch_size) as pbar:
            total_loss = 0
            for i in range(len(df) // batch_size):
                inp, out = [], []
                for i in range(batch_size):
                    i, o = next(train_iter)
                    inp.append(i)
                    out.append(o)
                inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.to(device)
                out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.to(device)

                meta_model.load_state_dict(model.state_dict())

                logits = meta_model(input_ids=inp_ids, labels=out_ids).logits
                ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))
                v = weights.forward(ce)
                weighted_ce = torch.sum(ce * v) / len(ce)
                weights.eta = weights.clamp(weights.eta)
                meta_model.zero_grad()
                grads = grad(weighted_ce, (meta_model.parameters()), create_graph=True)
                update_params(meta_model, lr=meta_lr, grads=grads)
                del grads

                logits = meta_model(input_ids=inp_ids, labels=out_ids).logits
                ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))
                weighted_ce = torch.sum(ce * v) / len(ce)
                grads_eta = grad(ce, weights.eta)
                weights.eta = weights.eta - meta_lr * grads_eta[0]
                del grads_eta

                eta = weights.clamp(weights.eta)

                logits = model(input_ids=inp_ids, labels=out_ids).logits
                ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))
                with torch.no_grad():
                    v = weights.forward(ce, eta)
                weighted_ce = torch.sum(ce * v) / len(ce)
                optimizer.zero_grad()
                weighted_ce.backward()
                optimizer.step()

                total_loss += weighted_ce.item()

                logs['eta'][epoch].append(eta.item())
                logs['loss'][epoch].append(weighted_ce.item())
                logs['zeros'][epoch].append(torch.sum(v == 0).item())
            
                pbar.update(1)
                pbar.set_postfix({'loss': total_loss / i+1})
        epoch += 1

    end = time.time() - start 
    model.save_pretrained(os.path.join(out_dir, 'model'))
    with open(os.path.join(out_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f)
    with open(os.path.join(out_dir, 'time.pkl'), 'wb') as f:
        pickle.dump(end, f)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)