from fire import Fire
import subprocess as sp
from os.path import join

def main(script, data, dataset, out_dir, batch_size=16, lr=0.001, wandb_project=None, sample=False):
    start = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for s in start:
        out = join(out_dir, f'paced_{s}')
        args = f'python {script} --data {data} --dataset_name {dataset} --out_dir {out} --batch_size {batch_size} --lr {lr}  --wandb_project {wandb_project}'
        if sample: args += ' --sample'
        sp.run(args, shell=True)

if __name__ == '__main__':
    Fire(main)