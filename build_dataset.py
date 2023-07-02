import pyterrier as pt 
pt.init()

from fire import Fire
from paced.util import *
import re
import ir_datasets
import pandas as pd

def main(dataset : str, terrier_dataset : str, out_dir : str, cut=0):
    ds = pt.get_dataset(terrier_dataset)
    indx = pt.IndexFactory.of(ds.get_index(variant='terrier_stemmed'))
    scorer = pt.batchretrieve.TextScorer(body_attr='text', wmodel='BM25', background_index=indx, properties={"termpipelines" : "Stopwords,PorterStemmer"})
    
    dataset = ir_datasets.load(dataset)
    triples = pd.DataFrame(dataset.docpairs_iter())

    new_df = collapse_triples(triples, scorer, dataset, num_docs=cut)
    new_df.to_json(out_dir, orient='records', lines=True, index=False)

if __name__ == '__main__':
    Fire(main)