from typing import Union, Optional

import mip
import numpy as np
import pytest

from halfspace import Model
from halfspace.convex_term import Func, FuncWithGrad, Grad, ConvexTerm, QueryPoint


def _process_callbacks(
    func: Func,
    grad: Grad,
    combine_grad: bool,
    approximate_grad: bool,
) -> tuple[Union[Func, FuncWithGrad], Optional[Union[Grad, bool]]]:
    if combine_grad and approximate_grad:
        raise ValueError
    if combine_grad:

        def func_with_grad(*args, **kwargs):
            return func(*args, **kwargs), grad(*args, **kwargs)

        return func_with_grad, True
    if approximate_grad:
        return func, None
    return func, grad


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
@pytest.mark.parametrize(
    ["combine_grad", "approximate_grad"], [(True, False), (False, True), (False, False)]
)
def test_single_variable_term(
    model: Model,
    query_point: dict[str, float],
    expected_value: float,
    expected_grad: float,
    combine_grad: bool,
    approximate_grad: bool,
):
    func, grad = _process_callbacks(
        func=lambda x: x**2,
        grad=lambda x: 2 * x,
        combine_grad=combine_grad,
        approximate_grad=approximate_grad,
    )
    term = ConvexTerm(
        var=model.var_by_name("x"),
        func=func,
        grad=grad,
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
@pytest.mark.parametrize(
    ["combine_grad", "approximate_grad"], [(True, False), (False, True), (False, False)]
)
def test_multivariable_term(
    model: Model,
    query_point: dict[str, float],
    expected_value: float,
    expected_grad: float,
    combine_grad: bool,
    approximate_grad: bool,
):
    func, grad = _process_callbacks(
        func=lambda x, y: x**2 + y**2,
        grad=lambda x, y: np.array([2 * x, 2 * y]),
        combine_grad=combine_grad,
        approximate_grad=approximate_grad,
    )
    term = ConvexTerm(
        var=(model.var_by_name("x"), model.var_by_name("y")),
        func=func,
        grad=grad,
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
@pytest.mark.parametrize(
    ["combine_grad", "approximate_grad"], [(True, False), (False, True), (False, False)]
)
def test_var_tensor_term(
    model: Model,
    query_point: dict[str, float],
    expected_value: float,
    expected_grad: float,
    combine_grad: bool,
    approximate_grad: bool,
):
    func, grad = _process_callbacks(
        func=lambda z: (z**2).sum(),
        grad=lambda z: 2 * z,
        combine_grad=combine_grad,
        approximate_grad=approximate_grad,
    )
    term = ConvexTerm(
        var=model.add_var_tensor(shape=(2,), lb=-10, ub=10, name="z"),
        func=func,
        grad=grad,
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
