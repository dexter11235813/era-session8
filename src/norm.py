import torch.nn as nn
from abc import ABC, abstractmethod
from functools import partial
from src import config


class NormFactory(ABC):
    @abstractmethod
    def get_norm(self):
        pass


class BatchNorm(NormFactory):
    def get_norm(self):
        return nn.BatchNorm2d


class GroupNorm(NormFactory):
    def get_norm(self):
        return partial(nn.GroupNorm, num_groups=config.GROUP_SIZE)


class LayerNorm(NormFactory):
    def get_norm(self):
        return partial(nn.GroupNorm, num_groups=1)
