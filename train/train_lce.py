from typing import NamedTuple
from fire import Fire
from pacednegatives.lceT5 import LCEModel, LCEDataModule, ChangeDifficulty
import os
import lightning.pytorch as pl

class HParams(NamedTuple):
    model_name : str 
    total_steps : int
    eta : float
    batch_size : int 
    lr : float 
    n : int
    warmup_steps : int 
    use_mean : bool 
    ignore_index : int 

def main(data: str, 
         dataset_name: str, 
         out_dir: str, 
         total_steps: int = 100000, 
         eta: float = 0.0, 
         batch_size: int = 16, 
         lr: float = 0.001, 
         meta_lr = None,
         var: float = 0.01, 
         n: int = 2, 
         use_max: bool = True, 
         warmup_steps: int = 10000,
         sample: bool = False, 
         use_mean: bool = True, 
         num_gpus: int = 1, 
         wandb_project: str = None):
    
    os.makedirs(out_dir, exist_ok=True)
    
    '''
    args = HParams(model_name='t5-base',
                     total_steps=total_steps,
                     eta=eta,
                     batch_size=batch_size,
                     lr=lr,
                     n=n,
                     warmup_steps=warmup_steps,
                     use_mean=use_mean,
                     ignore_index=-100)
    '''

    args ={
        'model_name': 't5-base',
        'total_steps': total_steps,
        'eta': eta,
        'batch_size': batch_size,
        'lr': lr,
        'meta_lr' : lr if not meta_lr else meta_lr,
        'n': n,
        'warmup_steps': warmup_steps,
        'use_mean': use_mean,
        'ignore_index': -100
    }

    model = LCEModel(args)
    # set up data module
    
    data_module = LCEDataModule(data, dataset_name, model.model.tokenizer, batch_size, sample, use_max, var=var, n=n)
    data_module.setup()
    
    # set up model
   

    logger = pl.loggers.WandbLogger(project=wandb_project)
    
    trainer_args = {
        'accelerator': 'gpu',
        'devices': num_gpus,
        'strategy': 'auto',
        'max_epochs': 1,
        'callbacks': [pl.callbacks.ProgressBar(), ChangeDifficulty(), pl.callbacks.LearningRateMonitor(logging_interval='step')],
        'logger': logger
    }

    # set up trainer
    trainer = pl.Trainer(**trainer_args)   
    
    # train
    trainer.fit(model, data_module)

    # save model
    model.model.model.save_pretrained(os.path.join(out_dir, 'model'))

if __name__ == '__main__':
    Fire(main)

