from fire import Fire
import subprocess as sp
from os.path import join
import numpy as np

def main(script, data, dataset, out_dir, batch_size=16, lr=0.001, wandb_project=None, sample=False, mean=False, num_gpus : int = 1):
    #variance = [0.01, 0.025, 0.05, 0.0075, 0.1]
    #N = [2, 4, 8, 16]
    variance = [0.01, 0.05, 0.1]
    N = [2, 4, 16]
    start = -np.log(0.5)*0.5

    for var in variance:
        for n in N:
            out = join(out_dir, f'paced_{var}_{n}')
            args = f'python {script} --data {data} --dataset_name {dataset} --out_dir {out} --batch_size {batch_size} --lr {lr} --wandb_project {wandb_project} --eta {start} --var {var} --n {n} --num_gpus {num_gpus}'
            if sample: args += ' --sample'
            if mean: args += ' --use_mean'
            sp.run(args, shell=True)

if __name__ == '__main__':
    Fire(main)