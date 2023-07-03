import pandas as pd
from fire import Fire
from paced.util import take_balanced_subset

def main(input_file, output_file, n):
    print(input_file)
    df = pd.read_json(input_file, orient='records', lines=True)
    df = take_balanced_subset(df, n)
    df.to_json(output_file, orient='records', lines=True)

if __name__ == '__main__':
    Fire(main)