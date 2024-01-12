from typing import Optional

import mip
import numpy as np
import pandas as pd
import pytest

from halfspace import Model
from halfspace.model import Var

VAR_TOL: float = 1e-2


def _check_solution(
    model: Model,
    expected_objective_value: Optional[float],
    expected_solution: Optional[dict[Var, float]],
    expected_status: mip.OptimizationStatus = mip.OptimizationStatus.OPTIMAL,
):
    if expected_objective_value is not None:
        assert model.objective_value == pytest.approx(
            expected_objective_value, abs=model.max_gap_abs
        )
    if expected_solution is not None:
        for x, expected_value in expected_solution.items():
            assert model.var_value(x=x) == pytest.approx(expected_value, abs=VAR_TOL)
    assert model.status == expected_status
    if expected_status == mip.OptimizationStatus.OPTIMAL:
        assert model.gap <= model.max_gap or model.gap_abs <= model.max_gap_abs
    if expected_status in (
        mip.OptimizationStatus.OPTIMAL,
        mip.OptimizationStatus.FEASIBLE,
    ):
        assert isinstance(model.search_log, pd.DataFrame)


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
        expected_objective_value=1 + 0.25**2,
        expected_solution={x: 0},
    )


def test_single_variable_binary():
    model = Model()
    x = model.add_var(var_type="B")
    model.add_objective_term(var=x, func=lambda x: (x - 0.25) ** 2 + 1)
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=1 + 0.25**2,
        expected_solution={x: 0},
    )


def test_multivariable_variable_no_constraints():
    model = Model()
    x = model.add_var(lb=0, ub=1)
    y = model.add_var(lb=0, ub=1)
    model.add_objective_term(
        var=(x, y), func=lambda x, y: (x - 0.25) ** 2 + (y - 0.25) ** 2 + 1
    )
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=1,
        expected_solution={x: 0.25, y: 0.25},
    )


def test_multivariable_variable_as_tensor_no_constraints():
    model = Model()
    x = model.add_var_tensor(shape=(2,), lb=0, ub=1)
    model.add_objective_term(
        var=x, func=lambda x: (x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2 + 1
    )
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
    model.add_objective_term(
        var=(x, y), func=lambda x, y: (x - 0.25) ** 2 + (y - 0.25) ** 2 + 1
    )
    model.add_linear_constr(100 * x + y <= 0.25)
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=1 + 0.25**2,
        expected_solution={x: 0.0, y: 0.25},
    )


def test_multivariable_linear_constraint_infeasible():
    model = Model()
    x = model.add_var(lb=0, ub=1)
    y = model.add_var(lb=0, ub=1)
    model.add_objective_term(
        var=(x, y), func=lambda x, y: (x - 0.25) ** 2 + (y - 0.25) ** 2 + 1
    )
    model.add_linear_constr(x + y >= 3)
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=None,
        expected_solution=None,
        expected_status=mip.OptimizationStatus.INFEASIBLE,
    )


def test_multivariable_nonlinear_constraint():
    model = Model(max_gap_abs=1e-2)
    x = model.add_var(lb=0, ub=1)
    y = model.add_var(lb=0, ub=1)
    model.add_objective_term(
        var=(x, y), func=lambda x, y: (x - 0.25) ** 2 + (y - 0.25) ** 2 + 1
    )
    model.add_nonlinear_constr(
        var=(x, y), func=lambda x, y: (80 * x) ** 2 + y**2 - 0.25**2
    )
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=1 + 0.25**2,
        expected_solution={x: 0.0, y: 0.25},
    )


def test_multivariable_nonlinear_constraint_infeasible():
    model = Model(max_gap_abs=1e-2)
    x = model.add_var(lb=0, ub=1)
    y = model.add_var(lb=0, ub=1)
    model.add_objective_term(
        var=(x, y), func=lambda x, y: (x - 0.25) ** 2 + (y - 0.25) ** 2 + 1
    )
    model.add_nonlinear_constr(var=(x, y), func=lambda x, y: np.exp(x + y) + 1)
    model.optimize()
    _check_solution(
        model=model,
        expected_objective_value=None,
        expected_solution=None,
        expected_status=mip.OptimizationStatus.INFEASIBLE,
    )
