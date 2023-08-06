import typing as t
from dataclasses import dataclass

from torch import Tensor

__all__ = ['ns_dataparser_indicator']

@dataclass(init=False)
class NerfStudioDatasetIndicator:
    train_indices: t.Sequence[int] = None
    eval_indices: t.Sequence[int] = None

    def set_train_indices(self, train_indices: Tensor):
        self.train_indices = train_indices.tolist()

    def set_eval_indices(self, eval_indices: Tensor):
        self.eval_indices = eval_indices.tolist()

    def is_initialized(self) -> bool:
        if self.train_indices is not None and self.eval_indices is not None:
            return True
        return False


ns_dataparser_indicator = NerfStudioDatasetIndicator()

