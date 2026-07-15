"""Easel Data Class

This module defines the :class:`Data` base class, which users subclass to
provide datasets and dataloaders for the supervised learning pipeline.

Public objects:
    Data: Base class for dataset preparation and dataloader construction.
"""

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Optional, Union


class Data:
    """Base class for preparing datasets and constructing dataloaders.

    Subclass :class:`Data` and override :meth:`prepare` (for downloads or
    preprocessing that should run once on the main process) and :meth:`setup`
    (to assign datasets per stage). The ``*_dataloader`` methods wrap the
    corresponding datasets in :class:`~torch.utils.data.DataLoader` instances
    and may be overridden for finer control.

    Attributes:
        train_dataset: Dataset used during training (or ``None``).
        val_dataset: Dataset used during validation (or ``None``).
        test_dataset: Dataset used during testing (or ``None``).
        predict_dataset: Dataset used during prediction (or ``None``).
    """

    def __init__(self) -> None:
        self.train_dataset: Optional[Union[Dataset, IterableDataset]] = None
        self.val_dataset: Optional[Union[Dataset, IterableDataset]] = None
        self.test_dataset: Optional[Union[Dataset, IterableDataset]] = None
        self.predict_dataset: Optional[Union[Dataset, IterableDataset]] = None

    def prepare(self) -> None:
        """Run downloads or preprocessing on the main process.

        Override this in subclasses to perform one-time work.
        The :class:`~easel.engine.Engine` calls this only on the main
        process before invoking :meth:`setup`.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Assign datasets for the given stage.

        Args:
            stage: The current stage (``"train"``, ``"eval"``,
            or ``None``). Override this to build and
            assign ``self.train_dataset``, ``self.val_dataset``, etc.
        """
        pass

    def train_dataloader(self, **kwargs) -> Union[DataLoader, None]:
        """Construct the training dataloader. Users can override this function
        for custom dataloader implementation.

        Returns ``None`` if ``self.train_dataset`` is not set.

        Args:
            **kwargs: Forwarded to :class:`~torch.utils.data.DataLoader`.

        Returns:
            A :class:`~torch.utils.data.DataLoader` wrapping the training
            dataset, or ``None`` if no training dataset is configured.
        """
        if self.train_dataset is None:
            return None
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self, **kwargs) -> Union[DataLoader, None]:
        """Construct the validation dataloader. Users can override this function
        for custom dataloader implementation.

        Returns ``None`` if ``self.val_dataset`` is not set.

        Args:
            **kwargs: Forwarded to :class:`~torch.utils.data.DataLoader`.

        Returns:
            A :class:`~torch.utils.data.DataLoader` wrapping the validation
            dataset, or ``None`` if no validation dataset is configured.
        """
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self, **kwargs) -> Union[DataLoader, None]:
        """Construct the testing dataloader. Users can override this function
        for custom dataloader implementation.

        Returns ``None`` if ``self.test_dataset`` is not set.

        Args:
            **kwargs: Forwarded to :class:`~torch.utils.data.DataLoader`.

        Returns:
            A :class:`~torch.utils.data.DataLoader` wrapping the test
            dataset, or ``None`` if no test dataset is configured.
        """
        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset, **kwargs)

    def predict_dataloader(self, **kwargs) -> Union[DataLoader, None]:
        """Construct the prediction dataloader. Users can override this function
        for custom dataloader implementation.

        Returns ``None`` if ``self.predict_dataset`` is not set.

        Args:
            **kwargs: Forwarded to :class:`~torch.utils.data.DataLoader`.

        Returns:
            A :class:`~torch.utils.data.DataLoader` wrapping the prediction
            dataset, or ``None`` if no prediction dataset is configured.
        """
        if self.predict_dataset is None:
            return None
        return DataLoader(self.predict_dataset, **kwargs)
