from fire import Fire
import pandas as pd

def main(pair_file : str, negative_file : str, output_file : str, cutoff : int = None):

    pairs = pd.read_json(pair_file, orient='records', dtype={'query_id': str, 'doc_id_a': str})
    negatives = pd.read_json(negative_file, orient='records', dtype={'query_id': str, 'doc_id_b': list})

    if cutoff is not None:
        pairs = pairs.sample(cutoff)
    
    negative_dict = negatives.set_index('query_id').doc_id_b.to_dict()

    pairs['doc_id_b'] = pairs['query_id'].apply(lambda x: negative_dict[x])
    pairs.to_json(output_file, orient='records')