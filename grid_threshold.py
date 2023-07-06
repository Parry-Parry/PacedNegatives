from fire import Fire
import subprocess as sp
from os.path import join

def main(script, data, dataset, out_dir, batch_size=16, lr=0.001, wandb_project=None):
    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
    step = [100, 1000, 10000, 100000000000]

    for success_threshold in thresholds:
        for heuristic_step_check in step:
            out = join(out_dir, f'paced_{success_threshold}_{heuristic_step_check}')
            sp.run(f'python {script} --data {data} --dataset_name {dataset} --out_dir {out} --batch_size {batch_size} --lr {lr} --success_threshold {success_threshold} --heuristic_step_check {heuristic_step_check} --wandb_project {wandb_project}', shell=True)

if __name__ == '__main__':
    Fire(main)