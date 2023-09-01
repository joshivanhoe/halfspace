import logging
from typing import Union, Iterable, Optional, Any, Type

import numpy as np


def _log_table_header(columns: Iterable[str]) -> None:
    columns = ["{:15}".format(col) for col in columns]
    line = '-{}-'.format("-".join("-" * len(col) for col in columns))
    logging.info(line)
    logging.info('|{}|'.format("|".join(columns)))
    logging.info(line)


def _log_table_row(values: Iterable[Union[float, int]]) -> None:
    values = [("{:15}" if isinstance(value, int) else "{:15.3e}").format(value) for value in values]
    logging.info('|{}|'.format("|".join(values)))


def _sigmoid(x: float, scale: float = 1.) -> float:
    return 1 / (1 + np.exp(-scale * x))


def _check_scalar(
    x: Any,
    name: str,
    var_type: Optional[Union[Type, tuple[TypeError]]] = None,
    lb: Optional[str] = None,
    ub: Optional[str] = None,
    include_boundaries: bool = True,
) -> None:
    pass
