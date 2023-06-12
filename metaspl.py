from collections import defaultdict
import json
import time
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
_K = 10e4
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
    def __init__(self, batches : int, batch_size : int, device = None, mu : float = 1.3, K : float = 10e4):
        super().__init__()
        self.v = nn.Parameter(torch.ones(batches, batch_size, requires_grad=True)).to(device)
        self.K = K
        self.mu = mu

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, batch_idx):
        assert batch_idx < self.v.shape[0]
        return gen_var(self.v[batch_idx], True)
    
    def set_weights(self, weights, batch_idx):
        self.v[batch_idx] = weights.to(self.device)
        
    def updateK(self):
        self.K = self.K * self.mu

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
         mu : float = 1.3,
         C : int = 5):
    
    logs = {
        'dataset': dataset,
        'model_name': model_name,
        'spl': True,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'meta_lr': meta_lr if meta_lr else lr,
        'cut': cut if cut else -1,
        'mu': mu,
        'loss' : defaultdict(list),
        'K' : defaultdict(list),
        'zeros' : defaultdict(list),
    }

    os.makedirs(out_dir, exist_ok=True)
    df  = process_dataset(ir_datasets.load(dataset), cut=cut)
    cut = len(df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_t5(model_name).to(device)
    meta_model = load_t5(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    optimizer = AdamW(model.parameters(), lr=lr)

    C = C / batch_size

    if not meta_lr: meta_lr = lr
    weights = Weights(ceil(len(df) / batch_size), batch_size * 2, device=device, mu=mu, K=_K / batch_size)

    def iter_train_samples():    
            while True:
                for row in df.itertuples():
                    yield 'Query: ' + row.query + ' Document: ' + row.pid + ' Relevant:', OUTPUTS[0]
                    yield 'Query: ' + row.query + ' Document: ' + row.nid + ' Relevant:', OUTPUTS[1]

    start = time.time()
    train_iter = _logger.pbar(iter_train_samples(), desc='total train samples')

    with _logger.pbar_raw(desc=f'train 1', total= cut // batch_size) as pbar:
        model.train()
        total_loss = 0
        count = 0
        for b in range(len(df) // batch_size):
            inp, out = [], []
            for i in range(batch_size):
                i, o = next(train_iter)
                inp.append(i)
                out.append(o)

            inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.to(device)
            out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.to(device)

            meta_model.load_state_dict(model.state_dict())

            logits = meta_model(input_ids=inp_ids, labels=out_ids).logits
            v = weights.forward(b)
            ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))
            weighted_ce = C * torch.sum(ce * v) / torch.sum(v)
            meta_model.zero_grad()
            grads = grad(weighted_ce, (meta_model.parameters()), create_graph=True)
            update_params(meta_model, lr=meta_lr, grads=grads)
            del grads

            logits = meta_model(input_ids=inp_ids, labels=out_ids).logits
            ce = torch.mean(loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))) - torch.sum(v) / weights.K
            grads_v = grad(ce, v)
            v_ce = ((v - meta_lr * grads_v[0]) < 0.5).type(torch.float32) 
            weights.set_weights(v_ce, b)
            del grads_v

            logits = model(input_ids=inp_ids, labels=out_ids).logits
            ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))
            with torch.no_grad():
                v = weights.forward(b)
            weighted_ce = torch.sum(ce * v) / torch.sum(v)
            optimizer.zero_grad()
            weighted_ce.backward()
            optimizer.step()

            logs['K'][0].append(weights.K)    
            logs['loss'][0].append(weighted_ce.item())
            logs['zeros'][0].append(torch.sum(v == 0).item())

            total_loss += weighted_ce.item()
            count += 1
            pbar.update(1)
            pbar.set_postfix({'loss': total_loss/count})
    
    logging.info('First Pass Complete')

    for epoch in range(1, epochs):
        with _logger.pbar_raw(desc=f'train {epoch}', total= cut // batch_size) as pbar:
            total_loss = 0
            count = 0
            for b in range(len(df) // batch_size):
                inp, out = [], []
                for i in range(batch_size):
                    i, o = next(train_iter)
                    inp.append(i)
                    out.append(o)
                inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.to(device)
                out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.to(device)

                meta_model.load_state_dict(model.state_dict())

                logits = meta_model(input_ids=inp_ids, labels=out_ids).logits
                v = weights.forward(b)
                logging.info('batch {b}: {v}'.format(b=b, v=v))
                ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))
                weighted_ce = C * torch.sum(ce * v) / torch.sum(v)
                meta_model.zero_grad()
                grads = grad(weighted_ce, (meta_model.parameters()), create_graph=True)
                update_params(meta_model, lr=meta_lr, grads=grads)
                del grads

                logits = meta_model(input_ids=inp_ids, labels=out_ids).logits
                ce = torch.mean(loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))) - torch.sum(v) / weights.K
                grads_v = grad(ce, v)
                v_ce = v_ce = ((v - meta_lr * grads_v[0]) < 0.5).type(torch.float32) 
                weights.set_weights(v_ce, b)
                del grads_v

                logits = model(input_ids=inp_ids, labels=out_ids).logits
                ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))
                with torch.no_grad():
                    v = weights.forward(b)
                weighted_ce = torch.sum(ce * v) / torch.sum(v)
                optimizer.zero_grad()
                weighted_ce.backward()
                optimizer.step()

                weights.updateK()
                total_loss += weighted_ce.item()

                logs['K'][epoch].append(weights.K)    
                logs['loss'][epoch].append(weighted_ce.item())
                logs['zeros'][epoch].append(torch.sum(v == 0).item())
            
                count += 1
                pbar.update(1)
                pbar.set_postfix({'loss': total_loss/count})
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