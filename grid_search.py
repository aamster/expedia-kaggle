from itertools import product
from typing import Dict


class GridSearch:
    """
    Class for performing grid search over hyperparameters
    """
    def __init__(self, grid: Dict[str, list]):
        self.grid = grid

    def iter_grid(self):
        """
        Yields a dictionary for every possible combination of hyperparameters.
        """
        keys = self.grid.keys()
        values = self.grid.values()
        for values in product(*values):
            yield dict(zip(keys, values))