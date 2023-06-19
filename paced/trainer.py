from typing import Any, Callable, Dict, List, Tuple
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments 

from .util import Weights

class PacedTrainer(Trainer):
    def __init__(self, 
                 # STD ARGS
                 model : PreTrainedModel | Any = None, 
                 args : TrainingArguments = None, 
                 data_collator : Any | None = None, 
                 train_dataset : Any | None = None, 
                 eval_dataset : Any | Dict[str, Any] | None = None, 
                 tokenizer : PreTrainedTokenizerBase | None = None, 
                 model_init : Callable[[], PreTrainedModel] | None = None, 
                 compute_metrics : Callable[[EvalPrediction], Dict] | None = None, 
                 callbacks : List[TrainerCallback] | None = None, 
                 optimizers : Tuple = ..., 
                 preprocess_logits_for_metrics: Callable[[Any, Any], Any] | None = None, 
                 # STD ARGS END
                 weights : Weights = None,
                 meta_optimizer : Tuple = ..., 
                 ):
        super().__init__(model, 
                         args, 
                         data_collator, 
                         train_dataset, 
                         eval_dataset, 
                         tokenizer, 
                         model_init, 
                         compute_metrics, 
                         callbacks, 
                         optimizers, 
                         preprocess_logits_for_metrics)

        self.meta_model = model.copy()
        self.meta_model.to(self.args.device)
        self.meta_optimizer = meta_optimizer
