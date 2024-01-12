from typing import Union

import mip
import numpy as np
import pytest

from halfspace import Model
from halfspace.convex_term import ConvexTerm, QueryPoint


def _check_convex_term(
    term: ConvexTerm,
    expected_value: float,
    expected_grad: Union[float, np.ndarray],
    expected_is_multivariable: bool,
    query_point: QueryPoint,
):
    # Check evaluation without gradient
    assert term(query_point=query_point, return_grad=False) == pytest.approx(
        expected_value
    )

    # Check evaluation with gradient
    value, grad = term(query_point=query_point, return_grad=True)
    assert value == pytest.approx(expected_value)
    assert grad == pytest.approx(expected_grad)

    # Check multivariable property
    assert term.is_multivariable == expected_is_multivariable

    # Check cut generation
    assert isinstance(term.generate_cut(query_point=query_point), mip.LinExpr)


@pytest.fixture()
def model() -> Model:
    m = Model()
    m.add_var(name="x", lb=-10, ub=10)
    m.add_var(name="y", lb=-10, ub=10)
    return m


@pytest.mark.parametrize(
    ["query_point", "expected_value", "expected_grad"],
    [
        ({"x": 0}, 0, 0),
        ({"x": 1}, 1, 2),
    ],
)
@pytest.mark.parametrize("approximate_grad", [True, False])
def test_single_variable_term(
    model: Model,
    query_point: dict[str, float],
    expected_value: float,
    expected_grad: float,
    approximate_grad: bool,
):
    term = ConvexTerm(
        var=model.var_by_name("x"),
        func=lambda x: x**2,
        grad=None if approximate_grad else lambda x: 2 * x,
    )
    _check_convex_term(
        term=term,
        expected_value=expected_value,
        expected_grad=expected_grad,
        expected_is_multivariable=False,
        query_point={
            model.var_by_name(name=name): value for name, value in query_point.items()
        },
    )


@pytest.mark.parametrize(
    ["query_point", "expected_value", "expected_grad"],
    [
        ({"x": 0, "y": 0}, 0, np.array([0, 0])),
        ({"x": 1, "y": 2}, 5, np.array([2, 4])),
    ],
)
@pytest.mark.parametrize("approximate_grad", [True, False])
def test_multivariable_term(
    model: Model,
    query_point: dict[str, float],
    expected_value: float,
    expected_grad: float,
    approximate_grad: bool,
):
    term = ConvexTerm(
        var=(model.var_by_name("x"), model.var_by_name("y")),
        func=lambda x_, y_: x_**2 + y_**2,
        grad=None if approximate_grad else lambda x_, y_: np.array([2 * x_, 2 * y_]),
    )
    _check_convex_term(
        term=term,
        expected_value=expected_value,
        expected_grad=expected_grad,
        expected_is_multivariable=True,
        query_point={
            model.var_by_name(name=name): value for name, value in query_point.items()
        },
    )


@pytest.mark.parametrize(
    ["query_point", "expected_value", "expected_grad"],
    [
        ({"z_0": 0, "z_1": 0}, 0, np.array([0, 0])),
        ({"z_0": 1, "z_1": 2}, 5, np.array([2, 4])),
    ],
)
@pytest.mark.parametrize("approximate_grad", [True, False])
def test_var_tensor_term(
    model: Model,
    query_point: dict[str, float],
    expected_value: float,
    expected_grad: float,
    approximate_grad: bool,
):
    z = model.add_var_tensor(shape=(2,), lb=-10, ub=10, name="z")
    term = ConvexTerm(
        var=z,
        func=lambda z_: z_[0] ** 2 + z_[1] ** 2,
        grad=None if approximate_grad else lambda z_: np.array([2 * z_[0], 2 * z_[1]]),
    )
    _check_convex_term(
        term=term,
        expected_value=expected_value,
        expected_grad=expected_grad,
        expected_is_multivariable=True,
        query_point={
            model.var_by_name(name=name): value for name, value in query_point.items()
        },
    )
