from time import time
from typing import Optional, Any

import mip
import numpy as np
import pandas as pd

from .utils import _log_table_header, _log_table_row

Start = list[tuple[mip.Var, float]]


class SearchState:

    def __init__(self, minimize: bool = True, logging_frequency: Optional[int] = 1):
        self.minimize = minimize
        self.logging_frequency = logging_frequency
        self._incumbent: float = (1 if self.minimize else -1) * mip.INF
        self._best: float = self._incumbent
        self._bound: float = -self._best
        self._start_time: float = time()
        self._time_elapsed: float = 0.
        self._iteration: int = 0
        self._iterations_without_improvement: int = 0
        self._log: list[dict[str, float]] = list()

    def update(self, incumbent: Optional[float] = None, bound: Optional[float] = None) -> None:

        # Update iteration and time elapsed
        self._iteration += 1
        self._time_elapsed = time() - self.start_time

        # Update incumbent and best solution
        if incumbent is not None:
            self._incumbent = incumbent
            if self.minimize == (incumbent < self.best):
                self._iterations_without_improvement = 0
                self._best = incumbent
            else:
                self._iterations_without_improvement += 1
        elif not np.isfinite(self._best):
            self._iterations_without_improvement += 1

        # Update bound
        if bound is not None:
            if self.minimize == (bound > self.bound):
                self._bound = bound

        # Update log
        self._log.append(self.to_dict())
        if self.logging_frequency is not None:
            if self.iteration % self.logging_frequency == 0:
                if self.iteration == 1:
                    _log_table_header(columns=self._log[-1].keys())
                _log_table_row(values=self._log[-1].values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "time_elapsed": self.time_elapsed,
            "incumbent": self.incumbent,
            "best": self.best,
            "bound": self.bound,
            "gap": self.gap,
        }

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def time_elapsed(self) -> float:
        return self._time_elapsed

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def iterations_without_improvement(self) -> int:
        return self._iterations_without_improvement

    @property
    def best(self) -> float:
        return self._best

    @property
    def incumbent(self) -> float:
        return self._incumbent

    @property
    def bound(self) -> float:
        return self._bound

    @property
    def gap(self) -> float:
        return abs(self.best - self.bound) / max(min(abs(self.best), abs(self.bound)), 1e-10)

    @property
    def log(self) -> pd.DataFrame:
        return pd.DataFrame(self._log)
