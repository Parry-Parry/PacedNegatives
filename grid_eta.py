from fire import Fire
import subprocess as sp
from os.path import join
import numpy as np

def main(script, data, dataset, out_dir, batch_size=16, lr=0.001, wandb_project=None, sample=False):
    start = [-np.log(0.5)*0.5, 0.1, 0.2, 0.3, 0.4, 0.5]
    meta_lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]

    for meta_lr in meta_lrs:
        for s in start:
            out = join(out_dir, f'paced_{s}_{meta_lr}')
            args = f'python {script} --data {data} --dataset_name {dataset} --out_dir {out} --batch_size {batch_size} --lr {lr} --meta_lr {meta_lr} --wandb_project {wandb_project} --eta {s}'
            if sample: args += ' --sample'
            sp.run(args, shell=True)

if __name__ == '__main__':
    Fire(main)