from models import *

def init_model(name : str):
    if name == 'monot5': 
        return init_t5()