import collections.abc
import numpy as np

from torch.utils.data import Dataset as _TorchDataset
from typing import Any, Callable, Optional, Sequence, Union
from torch.utils.data import Subset


class Dataset(_TorchDataset):
    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        self.data = data
        self.transform: Any = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        data_i = self.data[index]
        return self.transform(data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)
        return self._transform(index)


class UniformDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__(data=data, transform=transform)
        self.datasetkey = []
        self.data_dic = []
        self.datasetnum = []
        self.datasetlen = 0
        self.dataset_split(data)

    def dataset_split(self, data):
        keys = []
        for img in data:
            keys.append(img["image_name"].split("/")[0])
        self.datasetkey = list(np.unique(keys))
        data_dic = {}
        for iKey in self.datasetkey:
            data_dic[iKey] = [data[iSample] for iSample in range(len(keys)) if keys[iSample]==iKey]
        self.data_dic = data_dic
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the data {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(self.datasetkey)

    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return self.transform(data_i) if self.transform is not None else data_i

    def __getitem__(self, index):
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)