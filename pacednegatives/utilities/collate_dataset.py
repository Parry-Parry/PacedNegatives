from fire import Fire
import pandas as pd

def main(pair_file : str, negative_file : str, output_file : str, cutoff : int = None):

    pairs = pd.read_json(pair_file, orient='records')
    negatives = pd.read_json(negative_file, orient='records')

    if cutoff is not None:
        pairs = pairs.sample(cutoff)
    
    negative_dict = negatives.set_index('qid')['docno'].to_dict()

    pairs['doc_id_b'] = pairs['query_id'].apply(lambda x: negative_dict[x])
    pairs.to_json(output_file, orient='records')

if __name__ == '__main__':
    Fire(main)