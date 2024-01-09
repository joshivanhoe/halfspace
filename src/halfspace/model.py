import logging
from typing import Optional, Iterable, Union

import mip
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from .term import NonlinearTerm, Var, Fun, Grad
from .utils import check_scalar, log_table_header, log_table_row

Start = list[tuple[mip.Var, float]]


class Model:

    def __init__(
            self,
            minimize: bool = True,
            max_gap: float = 1e-4,
            max_gap_abs: float = 1e-4,
            feasibility_tol: float = 1e-4,
            step_size: float = 1e-6,
            smoothing: float = 0.9,
            solver_name: Optional[str] = None,
            log_freq: Optional[int] = 1,
    ):
        """

        Args:
            minimize: bool, default=`True`
            max_gap: float, default=1e-4
            max_gap_abs: float, default=1e-4
            feasibility_tol: float, default=1e-4
            step_size: float, default=1e-6
            solver_name: str or `None`, default=`None`
            log_freq: int or `None`, default = 1
        """
        self.minimize = minimize
        self.max_gap = max_gap
        self.max_gap_abs = max_gap_abs
        self.feasibility_tol = feasibility_tol
        self.step_size = step_size
        self.smoothing = smoothing
        self.solver_name = solver_name
        self.log_freq = log_freq
        self.reset()

    def reset(self) -> None:
        """Reset the model."""
        self._mip_model: mip.Model = mip.Model(
            solver_name=self.solver_name,
            sense=mip.MINIMIZE if self.minimize else mip.MAXIMIZE,
        )
        self._mip_model.verbose = 0
        self._mip_model.infeas_tol = self.feasibility_tol
        self._start: dict[mip.Var, float] = dict()
        self._objective_terms: list[NonlinearTerm] = list()
        self._nonlinear_constraints: list[NonlinearTerm] = list()
        self._best_solution: dict[mip.Var, float] = dict()
        self._best_objective: float = (1 if self.minimize else -1) * mip.INF
        self._best_bound: float = -self._best_objective
        self._search_log: list[dict[str, float]] = list()

    def add_var(
            self,
            lb: float,
            ub: float,
            var_type: str = mip.CONTINUOUS,
            name: str = ""
    ) -> mip.Var:
        """Add a decision variable to the model.

        Args:
            lb: float
            ub: float
            var_type: str, default='C'
            name: str, default=''
                The name of the decision variable

        Returns: mip.Var
            The decision variable.
        """
        lb, ub = self._validate_bounds(lb=lb, ub=ub, var_type=var_type)
        return self._mip_model.add_var(lb=lb, ub=ub, var_type=var_type, name=name)

    def add_variable_tensor(
            self,
            shape: tuple[int, ...],
            lb: float,
            ub: float,
            var_type: str = mip.CONTINUOUS,
            name: str = ""
    ) -> mip.LinExprTensor:
        """Add a tensor of decision variables to the model.

        Args:
            shape: tuple of int
                The shape of the tensor.
            lb: float=0.
            ub: float, default=inf
            var_type: str, default='C'
            name: str, default=''
                The name of the variable tensor.

        Returns: mip.LinExprTensor
            The tensor of decision variables.
        """
        lb, ub = self._validate_bounds(lb=lb, ub=ub, var_type=var_type)
        return self._mip_model.add_var_tensor(
            shape=shape,
            lb=lb,
            ub=ub,
            var_type=var_type,
            name=name,
        )

    def add_linear_constraint(self, constraint: mip.LinExpr, name: str = "") -> mip.Constr:
        """Add a linear constraint to the model.

        Args:
            constraint: mip.LinExpr
                The linear constraint.
            name: str, default=''
                The name of the constraint.

        Returns: mip.Constr
            The constraint
        """
        return self._mip_model.add_constr(lin_expr=constraint, name=name)

    def add_nonlinear_constraint(
            self,
            var: Var,
            func: Fun,
            grad: Optional[Grad] = None,
            name: str = "",
    ) -> NonlinearTerm:
        """Add a nonlinear constraint to the model.

        Args:
            var: mip.Var or list of mip.Var or mip.LinExprTensor
            func: callable
            grad: callable
            name: str, default=''

        Returns: NonlinearTerm
            The constraint
        """
        term = NonlinearTerm(
            var=var,
            func=func,
            grad=grad,
            feasibility_tol=self.feasibility_tol,
            step_size=self.step_size,
            is_constraint=True,
            name=name,
        )
        self._nonlinear_constraints.append(term)
        return term

    def add_objective_term(
            self,
            var: Var,
            func: Fun,
            grad: Optional[Grad] = None,
            name: str = "",
    ) -> NonlinearTerm:
        """Add an objective term to the model.

        Args:
            var: mip.Var or list of mip.Var or mip.LinExprTensor
            func: callable
            grad: callable
            name: str, default=''

        Returns: NonlinearTerm
            The objective term
        """
        term = NonlinearTerm(
            var=var,
            func=func,
            grad=grad,
            feasibility_tol=self.feasibility_tol,
            step_size=self.step_size,
            name=name,
        )
        self._objective_terms.append(term)
        return term

    def optimize(
            self,
            max_iters: int = 100,
            max_iters_no_improvement: Optional[int] = None,
            max_seconds_per_iter: Optional[float] = 100,
    ) -> mip.OptimizationStatus:

        # Define objective in epigraph form
        bound = self._mip_model.add_var(lb=-mip.INF, ub=mip.INF, name="_bound")
        self._mip_model.objective = bound

        # Initialize query point and search state
        query_point = {x: self._start.get(x) or (x.lb + x.ub) / 2 for x in self._mip_model.vars}

        iters_no_improvement = 0

        for i in range(max_iters):

            # Add objective cut
            expr = mip.xsum(term.generate_cut(query_point=query_point) for term in self.objective_terms)
            if self.minimize:
                self.add_linear_constraint(bound >= expr)
            else:
                self.add_linear_constraint(bound <= expr)

            # Re-optimize MIP model
            status = self._mip_model.optimize(max_seconds=max_seconds_per_iter)

            # If no solution is found, exit solve and return status
            if status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
                logging.info(f"Solve unsuccessful - exiting with optimization status: '{status.value}'.")
                return status

            # Parse solution
            solution = {var: var.x for var in self._mip_model.vars}
            objective = sum(term(solution) for term in self.objective_terms)
            self._best_bound = bound.x

            # Add cuts for violated nonlinear constraints
            is_feasible = True
            for constraint in self.nonlinear_constraints:
                expr = constraint.generate_cut(query_point=solution)
                if expr is not None:  # If the constraint is not violated, the cut expression will be `None`
                    is_feasible = False
                    self.add_linear_constraint(expr <= 0)

            # Update best solution/objective and query point
            if self.minimize == (objective < self.best_objective) and is_feasible:
                iters_no_improvement = 0
                self._best_objective = objective
                self._best_solution = solution
                query_point = solution
            else:
                if np.isfinite(self.best_objective):
                    iters_no_improvement += 1
                query_point = {
                    var: self.smoothing * query_point[var] + (1 - self.smoothing) * solution[var]
                    for var in self._mip_model.vars
                }

            # Update log
            self._search_log.append(
                {
                    "iteration": i,
                    "best_objective": self.best_objective,
                    "best_bound": self.best_bound,
                    "gap": self.gap,
                }
            )
            if self.log_freq is not None:
                if i == 0:
                    log_table_header(columns=self._search_log[-1].keys())
                if i % self.log_freq == 0:
                    log_table_row(values=self._search_log[-1].values())

            # Check early termination conditions
            if self.gap <= self.max_gap or self.gap_abs <= self.max_gap_abs:
                logging.info(f"Optimality tolerance reached - terminating search early.")
                return mip.OptimizationStatus.OPTIMAL
            if max_iters_no_improvement is not None:
                if iters_no_improvement > max_iters_no_improvement:
                    logging.info(f"Max iterations without improvement reached - terminating search early.")
                return mip.OptimizationStatus.FEASIBLE

        logging.info(f"Max iterations reached - terminating search.")
        return mip.OptimizationStatus.FEASIBLE

    @property
    def start(self) -> Start:
        return [(key, value) for key, value in self._start.items()]

    @start.setter
    def start(self, value: Start) -> None:
        # TODO add validation checks here
        self._start = {var: x for var, x in value}
        self._mip_model.start = value

    @property
    def objective_terms(self) -> list[NonlinearTerm]:
        return self._objective_terms

    @property
    def nonlinear_constraints(self) -> list[NonlinearTerm]:
        return self._nonlinear_constraints

    @property
    def best_solution(self) -> dict[mip.Var, float]:
        return self._best_solution

    @property
    def best_objective(self) -> float:
        return self._best_objective

    @property
    def best_bound(self) -> float:
        return self._best_bound

    @property
    def gap(self) -> float:
        return self.gap_abs / max(min(abs(self.best_objective), abs(self.best_bound)), 1e-10)

    @property
    def gap_abs(self) -> float:
        return abs(self.best_objective - self.best_bound)

    @property
    def search_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._search_log).set_index("iteration")

    @staticmethod
    def sum(terms: Iterable[Union[mip.Var, mip.LinExpr]]) -> mip.LinExpr:
        return mip.xsum(terms=terms)

    def plot_search(
            self,
            log_scale: bool = False,
            show_gap: bool = False,
            max_iters: Optional[int] = None
    ) -> Figure:
        df = self.search_log
        if max_iters is not None:
            df = df.iloc[:max_iters]
        return px.line(
            data_frame=df,
            y="gap" if show_gap else ["best_objective", "best_bound"],
            log_y=log_scale,
        )

    def _validate_params(self) -> None:
        check_scalar(x=self.smoothing, name="smoothing", lb=0., ub=1.)

    @staticmethod
    def _validate_bounds(lb: float, ub: float, var_type: str) -> tuple[float, float]:
        if var_type == mip.BINARY:
            lb, ub = 0, 1
        else:
            check_scalar(x=lb, name="lb", var_type=(float, int), ub=ub, lb=-mip.INF, include_boundaries=False)
            check_scalar(x=ub, name="ub", var_type=(float, int), ub=mip.INF, lb=lb, include_boundaries=False)
        return lb, ub


