from fire import Fire
import os
import json
import pandas as pd
import ir_datasets as irds
from pacednegatives.distill import TeacherLoader, MarginMSELoss, MonoT5Model
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import wandb

def main(
        triples_file : str, 
        teacher_file : str,
        dataset_name : str, 
        out_dir : str, 
        total_steps : int = 100000, 
        batch_size : int = 16, 
        lr : float = 0.001, 
        max=True,
        warmup_steps=0,
        shuffle=False,
        wandb_project=None,):

    os.makedirs(out_dir, exist_ok=True)

    if wandb_project is not None:
        wandb.init(project=wandb_project, config={
                'variant': triples_file.split('/')[-1],
                'dataset': dataset_name,
                'total_steps': total_steps,
                'batch_size': batch_size,
                'lr': lr,
                'max': max,
                'warmup_steps': warmup_steps,
            })

    corpus = irds.load(dataset_name)
    
    model = MonoT5Model.init()
    loader = TeacherLoader(teacher_file, triples_file, corpus, model.tokenizer, batch_size=batch_size, shuffle=shuffle)

    opt = AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps//batch_size, num_training_steps=total_steps//batch_size)

    loader.setup()
    model.train()

    for i in range(total_steps // batch_size):
        x, y = loader.get_batch(i)
        x.to(model.device)
        y.to(model.device)
        pred = model.forward(x)

        opt.zero_grad()
        loss = MarginMSELoss(pred, y)
        loss.backward()
        opt.step()
        sched.step()

        if wandb_project is not None:
            wandb.log({'loss': loss.item()})

        if i % 10 == 0:
            logging.info(f'Batch {i} Loss {loss.item()}')

    model.save_pretrained(os.path.join(out_dir, 'model'))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)