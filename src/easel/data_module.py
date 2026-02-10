import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Optional, Any, Union

class DataModule:
    """
    Standard interface for data loading.
    """

    def __init__(self, train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
                     val_dataset: Optional[Union[Dataset, IterableDataset]] = None,
                     test_dataset: Optional[Union[Dataset, IterableDataset]] = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def prepare_data(self):
        pass

    def setup(self):
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
