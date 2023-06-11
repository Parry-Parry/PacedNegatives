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

gen_param = lambda x, : nn.Parameter(torch.Tensor([x]))

RND = 42
_C  = 50
_K = 10e4 / _C
OUTPUTS = ['true', 'false']

def cross_entropy(logits, target):
    logprobs = torch.nn.functional.log_softmax(logits.view(logits.shape[0], -1), dim=1)
    batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)

    return batchloss

def process_dataset(dataset, cut=None):
    frame = pd.DataFrame(dataset.docpairs_iter())
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id').text.to_dict()

    frame['query'] = frame['query_id'].apply(lambda x: queries[x])
    frame['pid'] = frame['doc_id_a'].apply(lambda x: docs[x])
    frame['nid'] = frame['doc_id_b'].apply(lambda x: docs[x])
    if cut: frame = frame.sample(cut, random_state=RND) 
    return frame[['query', 'pid', 'nid']]

torch.manual_seed(RND)
_logger = ir_datasets.log.easy()

def main(dataset : str, 
         out_dir : str,
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
        'K' : defaultdict(list),
        'zeros' : defaultdict(list),
    }

    os.makedirs(out_dir, exist_ok=True)
    df  = process_dataset(ir_datasets.load(dataset), cut=cut)
    cut = len(df)
    v = nn.Parameter(torch.ones(ceil(cut / batch_size), batch_size * 2, requires_grad=True)).cuda()

    model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    optimizer = AdamW(model.parameters(), lr=lr)
    if not meta_lr: meta_lr = lr

    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')

    def iter_train_samples():    
            while True:
                for row in df.itertuples():
                    yield 'Query: ' + row.query + ' Document: ' + row.pid + ' Relevant:', OUTPUTS[0]
                    yield 'Query: ' + row.query + ' Document: ' + row.nid + ' Relevant:', OUTPUTS[1]

    start = time.time()
    train_iter = _logger.pbar(iter_train_samples(), desc='total train samples')

    K = gen_param(_K).cuda()
    mu = gen_param(mu).cuda()

    C = _C / batch_size
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

            inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.cuda()
            out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.cuda()

            if spl:
                logits = model(input_ids=inp_ids, labels=out_ids).logits

                logging.info('logits shape: %s', logits.shape)
                logging.info('number of zeros: %s', torch.sum(v[b] == 0).item())

                out_ids = out_ids.to(logits.device)
                ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))

                K_loss = torch.sum(v[b]) / K
                #ce = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), out_ids.view(-1), reduction='none')

                loss = (C / torch.sum(v[b])) * torch.sum(ce * v[b]) - K_loss
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

                    out_ids = out_ids.to(logits.device)
                    ce = loss_fct(logits.view(-1, logits.size(-1)), out_ids.view(-1))

                    K_loss = torch.sum(v[b]) / K
                    #ce = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), out_ids.view(-1), reduction='none')

                    loss = (C / torch.sum(v[b])) * torch.sum(ce * v[b]) - K_loss
                    grads = torch.autograd.grad(loss, v[b])
                    v[i] = nn.functional.sigmoid(v[i] - meta_lr * grads[0])
                    del grads
                    K = K * mu
                else:
                    loss = model(input_ids=inp_ids, labels=out_ids).loss
                
                logs['K'][epoch].append(K.item())    
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
    model.save_pretrained(os.path.join(out_dir, 'model'))
    with open(os.path.join(out_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f)
    with open(os.path.join(out_dir, 'time.pkl'), 'wb') as f:
        pickle.dump(end, f)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)