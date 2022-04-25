
from abc import ABCMeta, abstractmethod
import torch

class Attack(metaclass=ABCMeta):

    @abstractmethod
    def attack(self, x, y, verbose=1, EOT_size=1, EOT_batch_size=1):
        pass

    def compare(self, y, y_pred, targeted):
        if targeted:
            return (y_pred == y).tolist()
        else:
            return (y_pred != y).tolist()

        
