import logging
from typing import Union, Iterable, Optional, Any, Type


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
    columns = [f"{{:{width}}}".format(col) for col in columns]
    line = "-{}-".format("-".join("-" * len(col) for col in columns))
    logging.info(line)
    logging.info("|{}|".format("|".join(columns)))
    logging.info(line)


def log_table_row(values: Iterable[Union[float, int]], width: int = 15) -> None:
    """Log a table row.

    Logging level is set to `logging.INFO`.

    Args:
        values: iterable of str
            The values of the row.
        width: int, default=15
            The width of each column.

    Returns: None
    """
    values = [
        (f"{{:{width}}}" if isinstance(value, int) else f"{{:{width}.3e}}").format(
            value
        )
        for value in values
    ]
    logging.info("|{}|".format("|".join(values)))


def check_scalar(
    x: Any,
    name: str,
    var_type: Optional[Union[Type, tuple[Type, ...]]] = None,
    lb: Optional[Union[float, int]] = None,
    ub: Optional[Union[float, int]] = None,
    include_boundaries: bool = True,
) -> None:
    """Check that a scalar satisfies certain conditions.

    Args:
        x: Any
            The scalar to check.
        name: str,
            The name of the scalar. Used for error messages.
        var_type: type or tuple of types, default=`None`
            The expected type(s) of the scalar. If `None`, then no type checking is performed.
        lb: float or int, default=`None`
            The lower bound of the scalar. If `None`, then no lower bound checking is performed.
        ub: float or int, default=`None`
            The upper bound of the scalar. If `None`, then no upper bound checking is performed.
        include_boundaries: bool, default=`True`
            Whether to include the boundaries in the bound checking.

    Returns: None
    """
    if var_type is not None:
        assert isinstance(
            x, var_type
        ), f"Variable '{name}' ({type(x)}) is not expected type ({var_type})."
    if lb is not None:
        if include_boundaries:
            assert x >= lb, f"Variable '{name}' ({x}) is less than lower bound ({lb})."
        else:
            assert (
                x > lb
            ), f"Variable '{name}' ({x}) is less than or equal to lower bound ({lb})."
    if ub is not None:
        if include_boundaries:
            assert (
                x <= ub
            ), f"Variable '{name}' ({x}) is greater than lower bound ({ub})."
        else:
            assert (
                x < ub
            ), f"Variable '{name}' ({x}) is greater than or equal to lower bound ({ub})."
