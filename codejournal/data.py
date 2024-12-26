from torch.utils.data import DataLoader, Dataset
from .apply_func import apply_to_collection
from torch import Tensor
from abc import ABC

# https://github.com/Lightning-AI/pytorch-lightning/blob/0.9.0/pytorch_lightning/utilities/apply_func.py#L64-L89
class TransferableDataType(ABC):
    """
    A custom type for data that can be moved to a torch device via `.to(...)`.

    Example:

        >>> isinstance(dict, TransferableDataType)
        False
        >>> isinstance(torch.rand(2, 3), TransferableDataType)
        True
        >>> class CustomObject:
        ...     def __init__(self):
        ...         self.x = torch.rand(2, 2)
        ...     def to(self, device):
        ...         self.x = self.x.to(device)
        ...         return self
        >>> isinstance(CustomObject(), TransferableDataType)
        True
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is TransferableDataType:
            to = getattr(subclass, "to", None)
            return callable(to)
        return NotImplemented
        
def tensor_to_device(data, device):
    if isinstance(data, Tensor):
        data_output = data.to(device)
        return data_output
    return data

def batch_to(data, device):
    return apply_to_collection(data, dtype=TransferableDataType, function=tensor_to_device, device=device)

class SafeDataLoader(DataLoader):
    def __iter__(self):
        iterator = super().__iter__()
        while True:
            try:
                batch = next(iterator)
                yield batch
            except StopIteration:
                break
            except Exception as e:
                print(f"Error encountered in batch: {e}. Skipping batch and continuing.")
                continue  # Move to the next batch

__all__ = ["SafeDataLoader", "DataLoader", "Dataset","batch_to"]