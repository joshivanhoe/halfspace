"""Utility functions for the `halfspace` package."""

import logging
from typing import Iterable, Any



def log_table_header(columns: Iterable[str], width: int = 15) -> None:
    """Log a table header.

    Logging level is set to `logging.INFO`.

    Args:
        columns: iterable of str
            The column names of the table
        width: int, default=15
            The width of each column.

    Returns: None
    """
    columns_list = list(columns)
    if not columns_list:
        return
    
    columns = [f"{{:{width}}}".format(col) for col in columns_list]
    line = "-{}-".format("-".join("-" * len(col) for col in columns))
    logging.info(line)
    logging.info("|{}|".format("|".join(columns)))
    logging.info(line)


def log_table_row(values: Iterable[float | int], width: int = 15) -> None:
    """Log a table row.

    Logging level is set to `logging.INFO`.

    Args:
        values: iterable of float or int
            The values of the row.
        width: int, default=15
            The width of each column.

    Returns: None
    """
    values_list = list(values)
    if not values_list:
        return
        
    values_ = [(f"{{:{width}}}" if isinstance(value, int) else f"{{:{width}.3e}}").format(value) for value in values_list]
    logging.info("|{}|".format("|".join(values_)))


def check_scalar(
    x: Any,
    name: str,
    var_type: type | tuple[type, ...] | None = None,
    lb: float | int | None = None,
    ub: float | int | None = None,
    include_boundaries: bool = True,
) -> None:
    """Check that a scalar satisfies certain conditions.

    Args:
        x: Any
            The scalar to check.
        name: str,
            The name of the scalar. Used for error messages.
        var_type: type or tuple of types, default=None
            The expected type(s) of the scalar. If `None`, then no type checking is performed.
        lb: float or int, default=None
            The lower bound of the scalar. If `None`, then no lower bound checking is performed.
        ub: float or int, default=None
            The upper bound of the scalar. If `None`, then no upper bound checking is performed.
        include_boundaries: bool, default=True
            Whether to include the boundaries in the bound checking.

    Raises:
        ValueError: If the scalar does not meet the specified conditions.

    Returns: None
    """
    if var_type is not None:
        if not isinstance(x, var_type):
            raise ValueError(f"Variable '{name}' ({type(x)}) is not expected type ({var_type}).")
    if lb is not None:
        if include_boundaries:
            if x < lb:
                raise ValueError(f"Variable '{name}' ({x}) is less than lower bound ({lb}).")
        else:
            if x <= lb:
                raise ValueError(f"Variable '{name}' ({x}) is less than or equal to lower bound ({lb}).")
    if ub is not None:
        if include_boundaries:
            if x > ub:
                raise ValueError(f"Variable '{name}' ({x}) is greater than upper bound ({ub}).")
        else:
            if x >= ub:
                raise ValueError(f"Variable '{name}' ({x}) is greater than or equal to upper bound ({ub}).")
