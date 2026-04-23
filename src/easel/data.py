import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Optional, Union

class Data:
    def __init__(self):
        self.train_dataset: Optional[Union[Dataset, IterableDataset]] = None
        self.val_dataset: Optional[Union[Dataset, IterableDataset]] = None
        self.test_dataset: Optional[Union[Dataset, IterableDataset]] = None
        self.predict_dataset: Optional[Union[Dataset, IterableDataset]] = None

    def prepare(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, **kwargs) -> Union[DataLoader, None]:
        if self.train_dataset is None:
            return None
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self, **kwargs) -> Union[DataLoader, None]:
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self, **kwargs) -> Union[DataLoader, None]:
        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset, **kwargs)

    def predict_dataloader(self, **kwargs) -> Union[DataLoader, None]:
        if self.predict_dataset is None:
            return None
        return DataLoader(self.predict_dataset, **kwargs)
