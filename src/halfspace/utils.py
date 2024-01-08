import logging
from typing import Union, Iterable, Optional, Any, Type


def log_table_header(columns: Iterable[str], width: int = 15) -> None:
    columns = ["{:15}".format(col) for col in columns]
    line = '-{}-'.format("-".join("-" * len(col) for col in columns))
    logging.info(line)
    logging.info('|{}|'.format("|".join(columns)))
    logging.info(line)


def log_table_row(values: Iterable[Union[float, int]], width: int = 15) -> None:
    values = [("{:15}" if isinstance(value, int) else "{:15.3e}").format(value) for value in values]
    logging.info('|{}|'.format("|".join(values)))


def check_scalar(
    x: Any,
    name: str,
    var_type: Optional[Union[Type, tuple[Type, ...]]] = None,
    lb: Optional[Union[float, int]] = None,
    ub: Optional[Union[float, int]] = None,
    include_boundaries: bool = True,
) -> None:
    if var_type is not None:
        assert isinstance(x, var_type)
    if lb is not None:
        if include_boundaries:
            assert x >= lb
        else:
            assert x > lb
    if ub is not None:
        if include_boundaries:
            assert x <= ub
        else:
            assert x < ub
