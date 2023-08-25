from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

class MonoT5Model(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.rel = self.tokenizer.encode('true')[0]
        self.nrel = self.tokenizer.encode('false')[0]
    
    @staticmethod
    def init():
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        return MonoT5Model(model, tokenizer)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
    
    def forward(self, x):
        logits = self.model(**x).logits
        result = logits[:, 0, (self.REL, self.NREL)]
        return F.log_softmax(result, dim=1)[:, 0].cpu()
