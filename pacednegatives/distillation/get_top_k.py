import pyterrier as pt
if not pt.started():
    pt.init()
import os 
import pandas as pd
import numpy as np
import ir_datasets as irds
from collections import defaultdict

def collapse_doc_id_b(df):
    # take doc_id_a and append to doc_id_b which is a list of doc_ids
    df['all_docs'] = df.apply(lambda x: x['doc_id_b'].append(x['doc_id_a']), axis=1)
    return df

def collapse_teacher_scores(df):
    lookup = defaultdict(defaultdict(lambda : (0, 10000)))
    for row in df.itertuples():
        lookup[row.qid][row.doc_id] = (row.score, row.rank)
    return lookup

def main(
        triples_path,
        corpus_path,
        out_path,
):
    triples = pd.read_csv(triples_path, sep="\t", names=["qid", "doc_id_a", "doc_id_b"])
    corpus = irds.load(corpus_path)
    queries = pd.DataFrame(corpus.queries_iter()).set_index("qid").query.to_dict()
    teachers = {
        "BM25" : pt.BatchRetrieve(corpus, wmodel="BM25"),
        "TF_IDF" : pt.BatchRetrieve(corpus, wmodel="TF_IDF"),
        "PL2" : pt.BatchRetrieve(corpus, wmodel="PL2"),
    }

    all_queries = pd.DataFrame.from_records([{'qid' : qid, 'query' : queries[qid]} for qid in triples['qid'].unique()])

    all_lookup = {}
    for name, teacher in teachers.items():
        teacher_res = teacher.transform(all_queries)
        teacher_lookup = collapse_teacher_scores(teacher_res)
        all_lookup[name] = teacher_lookup
    
    

