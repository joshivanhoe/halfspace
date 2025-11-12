from contextlib import nullcontext as does_not_raise
from typing import Any, Iterable

import pytest

from halfspace.utils import check_param, log_table_header, log_table_row


@pytest.mark.parametrize("columns", [["a", "b", "c", "d"]])
@pytest.mark.parametrize("width", [10, 15])
def test_log_table_header(columns: Iterable[str], width: int):
    log_table_header(columns=columns, width=width)
    # TODO: add log checks


@pytest.mark.parametrize("values", [[1, 1.0, 2e10, 3e-10]])
@pytest.mark.parametrize("width", [10, 15])
def test_log_table_row(values: Iterable[float | int], width: int):
    log_table_row(values=values, width=width)
    # TODO: add log checks


@pytest.mark.parametrize(
    ["x", "name", "var_type", "lb", "ub", "include_boundaries", "expectation"],
    [
        (1, "x", int, 0, 2, True, does_not_raise()),
        (1, "x", None, 0, 2, True, does_not_raise()),
        (1, "x", int, None, 2, True, does_not_raise()),
        (1, "x", int, 0, None, True, does_not_raise()),
        (1, "x", int, 0, 2, False, does_not_raise()),
        (1, "x", float, 0, 2, True, pytest.raises(ValueError)),
        (1, "x", int, 2, 3, True, pytest.raises(ValueError)),
        (1, "x", int, 1, 2, False, pytest.raises(ValueError)),
        (1, "x", int, -1, 0, True, pytest.raises(ValueError)),
        (1, "x", int, 0, 1, False, pytest.raises(ValueError)),
    ],
)
def test_check_param(
    x: Any,
    name: str,
    var_type: type | tuple[type, ...] | None,
    lb: float | int | None,
    ub: float | int | None,
    include_boundaries: bool,
    expectation,
):
    with expectation:
        check_param(
            x=x,
            name=name,
            var_type=var_type,
            lb=lb,
            ub=ub,
            include_boundaries=include_boundaries,
        )
