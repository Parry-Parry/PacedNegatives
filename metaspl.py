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
import torch.backends.cudnn as cudnn
import os 
from math import ceil
import ir_datasets
import pickle
import pandas as pd
import logging

from torch.autograd import Variable, grad

gen_var = lambda x, y : Variable(x, requires_grad=y)

RND = 42
OUTPUTS = ['true', 'false']

loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')

def process_dataset(dataset, cut=None):
    frame = pd.DataFrame(dataset.docpairs_iter())
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id').text.to_dict()

    if cut: frame = frame.sample(cut, random_state=RND) 

    frame['query'] = frame['query_id'].apply(lambda x: queries[x])
    frame['pid'] = frame['doc_id_a'].apply(lambda x: docs[x])
    frame['nid'] = frame['doc_id_b'].apply(lambda x: docs[x])
    
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
    weight = lambda x, y : (-y/x) + 1
    def __init__(self, eta : float, device = None, min=np.log(2), max=10, tight=False):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_parameter(
            'eta_value',
            torch.nn.Parameter(
                torch.Tensor([eta]).to(self.device),
                requires_grad=True))
        self.clamp = lambda x : torch.clamp(x, min=min, max=max)
        self.eta = self.clamp(self.eta_value).requires_grad_()
        self.weighting = lambda x, y : (-x/y) + 1 
        self.forward = self.tight if tight else self.relaxed
        self.no_grad = self.no_grad_tight if tight else self.no_grad_relaxed

    def no_grad_relaxed(self, loss, eta):
        with torch.no_grad():
            weight = gen_var(torch.zeros(loss.size()), True).to(self.device)

            for i in range(len(loss)):
                if loss[i] > eta:
                    pass
                else:
                    weight[i]  = self.weighting(loss[i], eta)
            return weight

    def no_grad_tight(self, loss, eta):
        with torch.no_grad():
            weight = gen_var(torch.zeros(loss.size()), True).to(self.device)

            for i in range(len(loss)):
                if loss[i] > eta:
                    pass
                else:
                    weight[i] = torch.ones(1).to(self.device)
            return weight
    
    def relaxed(self, loss):
        weight = gen_var(torch.zeros(loss.size()), True).to(self.device)

        for i in range(len(loss)):
            if loss[i] > self.eta:
                weight[i] = torch.zeros(1).to(self.device).requires_grad_() * self.eta
            else:
                weight[i] = self.weighting(loss[i], self.eta)
        return weight
    
    def tight(self, loss):
        weight = gen_var(torch.zeros(loss.size()), True).to(self.device)

        for i in range(len(loss)):
            if loss[i] > self.eta:
                weight[i] = torch.zeros(1).to(self.device).requires_grad_() * self.eta 
            else:
                weight[i] = torch.ones(1).to(self.device).requires_grad_() / self.eta
        return weight
    
def adjust_lr(optimizer, init_lr, total_epochs):
    def update(epoch):
        lr = init_lr * ((0.2 ** int(epoch >= total_epochs * 1/4)) * (0.2 ** int(epoch >= total_epochs * 1/2))* (0.2 ** int(epoch >= total_epochs * 3/4)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return update
        

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
         min_eta : float = 2,
         max_eta : float = 10.0,
         tight : bool = False):
    
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
    cudnn.benchmark = True

    model = load_t5(model_name).to(device)
    meta_model = load_t5(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    optimizer = AdamW(model.parameters(), lr=lr)
    update_lr = adjust_lr(optimizer, lr, epochs)

    if not meta_lr: meta_lr = lr
    weights = Weights(eta, device=device, min=np.log(min_eta), max=max_eta, tight=tight)
    weights.eta = torch.tensor([eta], requires_grad=True).to(device)

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
            for j in range(len(df) // batch_size):
                inp, out = [], []
                for i in range(batch_size):
                    i, o = next(train_iter)
                    inp.append(i)
                    out.append(o)
                inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.to(device)
                out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.to(device)
                inp_ids = Variable(inp_ids, requires_grad=False)
                out_ids = Variable(out_ids, requires_grad=False)

                meta_model.load_state_dict(model.state_dict())

                logits = meta_model(input_ids=inp_ids, labels=out_ids).logits
                ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))
                weights.eta = weights.clamp(weights.eta)
                v = weights.forward(ce.data)

                weighted_ce = torch.sum(ce * v) / len(ce)
                meta_model.zero_grad()
                grads = grad(weighted_ce, (meta_model.parameters()), create_graph=True, retain_graph=True)
                meta_lr = lr * ((0.1 ** int(epoch >= epochs * 1/4)) * (0.1 ** int(epoch >= epochs * 1/2)))
                update_params(meta_model, lr=meta_lr, grads=grads)
                del grads

                logits = meta_model(input_ids=inp_ids, labels=out_ids).logits
                ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))
                weighted_ce = torch.sum(ce * v) / len(ce)
                grads_eta = grad(weighted_ce, weights.eta)
                weights.eta = weights.eta - meta_lr * grads_eta[0]
                del grads_eta

                eta = weights.clamp(weights.eta)

                logits = model(input_ids=inp_ids, labels=out_ids).logits
                ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))
                v = weights.no_grad(ce, eta)
                weighted_ce = torch.sum(ce * v) / len(ce)
                optimizer.zero_grad()
                weighted_ce.backward()
                optimizer.step()

                if j == 1: 
                    logging.info(f'loss: {ce.item()} | v : {v}')

                total_loss += weighted_ce.item()

                if j % 100 == 0:
                    logging.info(f'BATCH: {j} | Average v: {torch.mean(v).item()} | eta: {eta.item()}')

                logs['eta'][epoch].append(eta.item())
                logs['loss'][epoch].append(weighted_ce.item())
                logs['zeros'][epoch].append(torch.sum(v == 0).item())
            
                pbar.update(1)
                pbar.set_postfix({'loss': total_loss / (j+1)})

        update_lr(epoch)
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