from fire import Fire
import subprocess as sp
from os.path import join
import numpy as np

def main(script, 
         data, 
         dataset, 
         out_dir, 
         batch_size=16, 
         lr=0.001, 
         meta_lr=None, 
         wandb_project=None, 
         sample=False, 
         mean=False, 
         num_gpus : int = 1,
         num_workers : int = None,
         max_steps : int = None):
    #variance = [0.01, 0.025, 0.05, 0.0075, 0.1]
    #N = [2, 4, 8, 16]
    variance = [0.01, 0.05, 0.1]
    N = [4, 8]
    start = -np.log(0.5)*0.5

    for var in variance:
        for n in N:
            out = join(out_dir, f'paced_{var}_{n}')
            args = f'python {script} --data {data} --dataset_name {dataset} --out_dir {out} --batch_size {batch_size} --lr {lr} --wandb_project {wandb_project} --eta {start} --var {var} --n {n} --num_gpus {num_gpus}'
            if sample: args += ' --sample'
            if mean: args += ' --use_mean'
            if meta_lr: args += f' --meta_lr {meta_lr}'
            if num_workers: args += f' --num_workers {num_workers}'
            if max_steps: args += f' --max_steps {max_steps}'
            sp.run(args, shell=True)

if __name__ == '__main__':
    Fire(main)