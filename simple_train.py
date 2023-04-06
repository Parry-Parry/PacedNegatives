import fire 
import torch 
import os 
from dataloaders.spldataloader import irdsDataset
from loss import splloss, stdloss

def main(dataset : str, model_name : str, spl : bool = True, epochs : int = 1, batch_size : int = 16):
    train_data  = irdsDataset(f'{dataset}/train')

    loader = None

    if spl: loss_func = splloss
    else: loss_func = stdloss

    model = init_model(model_name)

    optimizer = None 
    if spl: v_optimizer = None 

    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            pass
            




if __name__=='__main__':
    fire.Fire(main)