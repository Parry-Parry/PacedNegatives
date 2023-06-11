from collections import defaultdict
import json
import time
import fire 
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW
import torch
import torch.nn as nn
import os 
from math import ceil
import ir_datasets
import pickle
import pandas as pd
import logging

RND = 42
_C  = 50
_K = 10e4 / _C
OUTPUTS = ['true', 'false']

def process_dataset(dataset, cut=None):
    frame = pd.DataFrame(dataset.docpairs_iter())
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(ir_datasets.load(dataset).queries_iter()).set_index('query_id').text.to_dict()

    frame['query'] = frame['query_id'].apply(lambda x: queries[x])
    frame['pid'] = frame['pos_doc_id'].apply(lambda x: docs[x])
    frame['nid'] = frame['neg_doc_id'].apply(lambda x: docs[x])
    if cut: frame = frame.sample(cut, random_state=RND) 
    return frame[['query', 'pid', 'nid']]

torch.manual_seed(RND)
_logger = ir_datasets.log.easy()

def main(dataset : str, 
         out : str,
         model_name : str = 't5-base',
         spl=False,
         epochs : int = 10, 
         batch_size : int = 128,
         lr : float = 5e-5,
         meta_lr : float = None,
         cut : int = None,
         mu : float = 1.3):
    
    logs = {
        'dataset': dataset,
        'model_name': model_name,
        'spl': spl,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'meta_lr': meta_lr if meta_lr else lr,
        'cut': cut if cut else -1,
        'mu': mu,
        'loss' : defaultdict(list),
        'K' : defaultdict(list) if spl else {},
        'zeros' : defaultdict(list) if spl else {},
    }

    os.makedirs(out, exist_ok=True)
    
    dataset = ir_datasets.load(dataset)
    df  = process_dataset(dataset, cut=cut)
    cut = len(df) * 2
    v = nn.parameter.Parameter(torch.ones(ceil(cut / batch_size), batch_size)).cuda()

    model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    optimizer = AdamW(model.parameters(), lr=lr)
    if not meta_lr: meta_lr = lr

    def iter_train_samples():    
            while True:
                for row in df.itertuples():
                    yield 'Query: ' + row.query + ' Document: ' + row.pid + ' Relevant:', OUTPUTS[0]
                    yield 'Query: ' + row.query + ' Document: ' + row.nid + ' Relevant:', OUTPUTS[1]

    start = time.time()
    train_iter = _logger.pbar(iter_train_samples(), desc='total train samples')

    K = nn.parameter.Parameter(_K).cuda()
    mu = nn.parameter.Parameter(mu).cuda()

    C = _C / batch_size
    with _logger.pbar_raw(desc=f'train {epoch}', total= cut // batch_size) as pbar:
        model.train()
        total_loss = 0
        count = 0
        for b in range(len(df) // batch_size):
            inp, out = [], []
            for i in range(batch_size):
                i, o = next(train_iter)
                inp.append(i)
                out.append(o)

            inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.cuda()
            out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.cuda()

            if spl:
                logits = model(input_ids=inp_ids, labels=out_ids).logits
                loss = (C / torch.sum(v[b])) * torch.sum(nn.functional.cross_entropy(logits, out_ids, reduction='none') * v[b]) - torch.sum(v[b]) / K
                grads = torch.autograd.grad(loss, v[b])
                v[i] = nn.functional.sigmoid(v[i] - meta_lr * grads[0])
                del grads
            else:
                loss = model(input_ids=inp_ids, labels=out_ids).loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss = loss.item()
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
                inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.cuda()
                out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.cuda()

                if spl:
                    logits = model(input_ids=inp_ids, labels=out_ids).logits
                    loss = (C / torch.sum(v[b])) * torch.sum(nn.functional.cross_entropy(logits, out_ids, reduction='none') * v[b]) - torch.sum(v[b]) / K
                    grads = torch.autograd.grad(loss, v[b])
                    v[i] = nn.functional.sigmoid(v[i] - meta_lr * grads[0])
                    del grads
                    K = K * mu
                else:
                    loss = model(input_ids=inp_ids, labels=out_ids).loss
                
                logs['K']['epoch'].append(K.item())    
                logs['loss'][epoch].append(loss.item())
                logs['zeros'][epoch].append(torch.sum(v[b] == 0).item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss = loss.item()
                count += 1
                pbar.update(1)
                pbar.set_postfix({'loss': total_loss/count})
        epoch += 1

    end = time.time() - start 
    model.save_pretrained(os.path.join(out, 'model'))
    with open(os.path.join(out, 'logs.json'), 'w') as f:
        json.dump(logs, f)
    with open(os.path.join(out, 'time.pkl'), 'wb') as f:
        pickle.dump(end, f)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)