import logging
from typing import NamedTuple
from fire import Fire
from pacednegatives.lceT5 import LCEModel, LCEDataModule, ChangeDifficulty
import os
import lightning.pytorch as pl
import lightning as L

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
         use_max: bool = False, 
         warmup_steps: int = 10000,
         sample: bool = False, 
         use_mean: bool = False, 
         num_gpus: int = 1, 
         wandb_project: str = None):
    #pl.seed_everything(42, workers=True)
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
        'total_steps': total_steps//batch_size,
        'eta': eta,
        'batch_size': batch_size,
        'lr': lr,
        'meta_lr' : lr if not meta_lr else meta_lr,
        'n': n,
        'warmup_steps': warmup_steps//batch_size,
        'use_mean': use_mean,
        'ignore_index': -100
    }

    model = LCEModel(args)
    # set up data module
    
    data_module = LCEDataModule(data, dataset_name, model.tokenizer, batch_size, sample, use_max, var=var, n=n, init_weight=model.weights.eta.item(), num_workers=4 if num_gpus < 2 else 0)
    data_module.setup()
    
    # set up model
   

    logger = pl.loggers.WandbLogger(project=wandb_project)
    
    trainer_args = {
        'devices' : num_gpus,
        'callbacks': [pl.callbacks.ProgressBar(), ChangeDifficulty(), pl.callbacks.LearningRateMonitor(logging_interval='step')],
        #'callbacks': [pl.callbacks.ProgressBar(), pl.callbacks.LearningRateMonitor(logging_interval='step')],
        'logger': logger,
        #'detect_anomaly' : True
        'max_epochs' : 1,
        'default_root_dir' : out_dir,
    }

    if num_gpus > 1:
        trainer_args['strategy'] = 'ddp'

    # set up trainer
    trainer = L.Trainer(**trainer_args)   

    #tuner = pl.tuner.Tuner(trainer)
    #tuner.scale_batch_size(model, mode="power", datamodule=data_module)
    
    # train
    trainer.fit(model, data_module)

    # save model
    model.model.save_pretrained(os.path.join(out_dir, 'finalmodel'))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)

