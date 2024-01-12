import mip
import pytest

from halfspace import Model
from halfspace.model import Var

VAR_TOL: float = 1e-2


def _check_solution(
    model: Model,
    expected_objective_value: float,
    expected_solution: dict[Var, float],
    expected_status: mip.OptimizationStatus = mip.OptimizationStatus.OPTIMAL,
):
    assert model.status == expected_status
    assert model.objective_value == pytest.approx(expected_objective_value, abs=model.max_gap_abs)
    for x, expected_value in expected_solution.items():
        assert model.var_value(x=x) == pytest.approx(expected_value, abs=VAR_TOL)


def test_single_variable_no_constraints():
    model = Model()
    x = model.add_var(lb=0, ub=1)
    model.add_objective_term(var=x, func=lambda x: (x - 0.25) ** 2 + 1)
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=1,
        expected_solution={x: 0.25},
    )


def test_single_variable_integer():
    model = Model()
    x = model.add_var(var_type="I", lb=0, ub=1)
    model.add_objective_term(var=x, func=lambda x: (x - 0.25) ** 2 + 1)
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=1 + 0.25 ** 2,
        expected_solution={x: 0},
    )


def test_single_variable_binary():
    model = Model()
    x = model.add_var(var_type="B")
    model.add_objective_term(var=x, func=lambda x: (x - 0.25) ** 2 + 1)
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=1 + 0.25 ** 2,
        expected_solution={x: 0},
    )


def test_multivariable_variable_no_constraints():
    model = Model()
    x = model.add_var(lb=0, ub=1)
    y = model.add_var(lb=0, ub=1)
    model.add_objective_term(var=(x, y), func=lambda x, y: (x - 0.25) ** 2 + (y - 0.25) ** 2 + 1)
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=1,
        expected_solution={x: 0.25, y: 0.25},
    )


def test_multivariable_variable_as_tensor_no_constraints():
    model = Model()
    x = model.add_var_tensor(shape=(2,), lb=0, ub=1)
    model.add_objective_term(var=x, func=lambda x: (x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2 + 1)
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=1,
        expected_solution={x[0]: 0.25, x[1]: 0.25},
    )


def test_multivariable_linear_constraint():
    model = Model()
    x = model.add_var(lb=0, ub=1)
    y = model.add_var(lb=0, ub=1)
    model.add_objective_term(var=(x, y), func=lambda x, y: (x - 0.25) ** 2 + (y - 0.25) ** 2 + 1)
    model.add_linear_constr(100 * x + y <= 0.25)
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=1 + 0.25 ** 2,
        expected_solution={x: 0., y: 0.25},
    )


def test_multivariable_nonlinear_constraint():
    model = Model(max_gap_abs=1e-2)
    x = model.add_var(lb=0, ub=1)
    y = model.add_var(lb=0, ub=1)
    model.add_objective_term(var=(x, y), func=lambda x, y: (x - 0.25) ** 2 + (y - 0.25) ** 2 + 1)
    model.add_nonlinear_constr(var=(x, y), func=lambda x, y: (80 * x) ** 2 + y ** 2 - 0.25 ** 2)
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=1 + 0.25 ** 2,
        expected_solution={x: 0., y: 0.25},
    )
