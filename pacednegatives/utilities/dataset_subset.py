import pandas as pd
from fire import Fire

def main(input_file : str, output_file : str, n):
    df = pd.read_json(input_file, orient='records')
    df = df.sample(n=n)
    df.to_json(output_file, orient='records')

if __name__ == '__main__':
    Fire(main)