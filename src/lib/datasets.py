from tinygrad import Tensor
from tinygrad.nn import datasets
from typing import Literal, Union


Dataset = tuple[Tensor, Tensor, Tensor, Tensor]


CustomDatasets = Literal["none"]
BuiltinDatasets = Literal["mnist", "cifar"]
Datasets = Union[BuiltinDatasets, CustomDatasets]


def load_custom_dataset(dataset: CustomDatasets, device=None) -> Dataset:
    pass


def load_dataset(dataset: Datasets, device=None) -> Dataset:
    if dataset == "mnist":
        return datasets.mnist(device)
    elif dataset == "cifar":
        return datasets.cifar(device)

    return load_custom_dataset(dataset, device)
