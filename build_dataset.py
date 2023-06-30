import pyterrier as pt 
pt.init()

from fire import Fire
from paced.util import *
import re
import ir_datasets
import pandas as pd

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return re.sub(r'/[^\x00-\x7F]/g', '', text).strip()

def main(dataset : str, out_dir : str, cut=0):
    ds = pt.get_dataset(dataset)
    indx = pt.IndexFactory.of(ds.get_index(variant='terrier_stemmed'))
    scorer = pt.batchretrieve.TextScorer(body_attr='text', wmodel='BM25', background_index=indx, properties={"termpipelines" : "Stopwords,PorterStemmer"})
    
    dataset = ir_datasets.load(dataset)
    triples = pd.DataFrame(dataset.doc_pairs_iter())

    new_df = collapse_triples(triples, scorer, dataset, num_docs=cut)
    new_df.to_json(out_dir, orient='records', lines=True, index=False)

if __name__ == '__main__':
    Fire(main)