from abc import ABC, abstractmethod
from torch.utils.data import TensorDataset, DataLoader
import torch



class MusicDataset():
    """
    Abstract Base Class for music data sets
    Must return
    """

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name # now only support dataset_name = 'nottingham', using other names will cause an error
        self._tensor_dataset = None


    @abstractmethod
    def parse_abc_txt(self):
        """
        return: Iterator over the dataset
        """
        pass
