from torch.data.util import Dataset
import ir_datasets

class irdsDataset(Dataset):
    def __init__(self, irds_name : str, **kwargs) -> None:
        super.__init__(**kwargs)
        ds = ir_datasets.load(irds_name)


