from collections import namedtuple
from fire import Fire
from pacednegatives.lceT5 import lceModel, LCEDataModule, ChangeDifficulty
from pacednegatives.dataloader import LCEDataset, LCELoader
import os
import lighting as L
import lightning.pytorch as pl

def main(data : str, 
        dataset_name : str, 
        out_dir : str, 
        total_steps : int = 100000, 
        eta : float = 0.0,
        batch_size : int = 16, 
        lr : float = 0.001, 
        var : float = 0.01,
        n : int = 2,
        use_max=True,
        warmup_steps=0,
        sample=False,
        use_mean=True,
        wandb_project=None,):
    
    os.makedirs(out_dir, exist_ok=True)

    hparams = namedtuple(
        'hparams',
        [
            ('total_steps', total_steps),
            ('eta', eta),
            ('batch_size', batch_size),
            ('lr', lr),
            ('var', var),
            ('n', n),
            ('max', max),
            ('warmup_steps', warmup_steps),
            ('sample', sample),
            ('use_mean', use_mean),
            ('wandb_project', wandb_project),
        ]
    )

    # set up wandb and pl trainer 

    # set up data module
    data_module = LCEDataModule(data, dataset_name, batch_size, sample, use_max)
    data_module.setup()
    
    # set up model
    model = lceModel(hparams)
    model.setup()
    
    # set up trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1,
        progress_bar_refresh_rate=20,
        callbacks=[pl.callbacks.ProgressBar(), ChangeDifficulty()],
    )   
    
    # train
    trainer.fit(model, data_module)

if __name__ == '__main__':
    Fire(main)

