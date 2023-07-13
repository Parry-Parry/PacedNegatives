from fire import Fire
import subprocess as sp
from os.path import join

def main(script, data, dataset, out_dir, batch_size=16, lr=0.001, wandb_project=None, sample=False):
    start_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    end_vals = [0.6, 0.7, 0.8, 0.9, 1.0]
    fracs = [0.1, 0.2, 0.3, 0.4, 0.5]

    for start_val in start_vals:
        for end_val in end_vals:
            for frac in fracs:
                out = join(out_dir, f'paced_{start_val}_{end_val}_{frac}')
                args = f'python {script} --data {data} --dataset_name {dataset} --out_dir {out} --batch_size {batch_size} --lr {lr} --start_difficulty {start_val} --max_difficulty {end_val} --frac_interpolat {frac} --wandb_project {wandb_project}'
                if sample: args += ' --sample'
                sp.run(args, shell=True)

if __name__ == '__main__':
    Fire(main)