import logging
from typing import Optional, Iterable, Union

import mip
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from .convex_term import ConvexTerm, Var, Fun, Grad
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
            smoothing: Optional[float] = .5,
            solver_name: Optional[str] = None,
            log_freq: Optional[int] = 1,
    ):
        """Model constructor.

        Args:
            minimize: bool, default=`True`
                Whether the objective should be minimized. If `False`, the objective will be maximized.
            max_gap: float, default=1e-4

            max_gap_abs: float, default=1e-4

            feasibility_tol: float, default=1e-4

            step_size: float, default=1e-6
                The step size used to numerically evaluate gradients

            smoothing: float, default=.5
                The

            solver_name: str or `None`, default=`None`
                The MIP solver to use. Valid options. If `None`, the default solver will be selected.
            log_freq: int or `None`, default = 1
                The frequency with which logs are
        """
        self.minimize = minimize
        self.max_gap = max_gap
        self.max_gap_abs = max_gap_abs
        self.feasibility_tol = feasibility_tol
        self.step_size = step_size
        self.smoothing = smoothing
        self.solver_name = solver_name
        self.log_freq = log_freq
        self._validate_params()
        self.reset()

    def reset(self) -> None:
        """Reset the model."""
        self._model: mip.Model = mip.Model(
            solver_name=self.solver_name,
            sense=mip.MINIMIZE if self.minimize else mip.MAXIMIZE,
        )
        self._model.verbose = 0
        self._model.infeas_tol = self.feasibility_tol
        self._start: dict[mip.Var, float] = dict()
        self._objective_terms: list[ConvexTerm] = list()
        self._nonlinear_constraints: list[ConvexTerm] = list()
        self._best_solution: dict[mip.Var, float] = dict()
        self._best_objective: float = (1 if self.minimize else -1) * mip.INF
        self._best_bound: float = -self._best_objective
        self._search_log: list[dict[str, float]] = list()

    def add_var(
            self,
            lb: Optional[float] = None,
            ub: Optional[float] = None,
            var_type: str = mip.CONTINUOUS,
            name: str = ""
    ) -> mip.Var:
        """Add a decision variable to the model.

        Args:
            lb: float
                The lower bound for the decision variable. Must be finite and less than the upper bound.
            ub: float
                The upper bound for the decision variable. Must be finite and greater than the lower bound.
            var_type: str, default='C'
                The variable type. Valid options are 'C' (continuous), 'I' (integer) and 'B' (binary).
            name: str, default=''
                The name of the decision variable.

        Returns: mip.Var
            The decision variable.
        """
        lb, ub = self._validate_bounds(lb=lb, ub=ub, var_type=var_type)
        return self._model.add_var(lb=lb, ub=ub, var_type=var_type, name=name)

    def add_var_tensor(
            self,
            shape: tuple[int, ...],
            lb: Optional[float] = None,
            ub: Optional[float] = None,
            var_type: str = mip.CONTINUOUS,
            name: str = ""
    ) -> mip.LinExprTensor:
        """Add a tensor of decision variables to the model.

        Args:
            shape: tuple of int
                The shape of the tensor.
            lb: float
                The lower bound for the decision variables. Must be finite and less than the upper bound.
            ub: float
                The upper bound for the decision variables. Must be finite and greater than the lower bound.
            var_type: str, default='C'
                The variable type. Valid options are 'C' (continuous), 'I' (integer) and 'B' (binary).
            name: str, default=''
                The name of the decision variable.

        Returns: mip.LinExprTensor
            The tensor of decision variables.
        """
        lb, ub = self._validate_bounds(lb=lb, ub=ub, var_type=var_type)
        return self._model.add_var_tensor(
            shape=shape,
            lb=lb,
            ub=ub,
            var_type=var_type,
            name=name,
        )

    def add_linear_constr(self, constraint: mip.LinExpr, name: str = "") -> mip.Constr:
        """Add a linear constraint to the model.

        Args:
            constraint: mip.LinExpr
                The linear constraint.
            name: str, default=''
                The name of the constraint.

        Returns: mip.Constr
            The constraint expression.
        """
        return self._model.add_constr(lin_expr=constraint, name=name)

    def add_nonlinear_constr(
            self,
            var: Var,
            func: Fun,
            grad: Optional[Grad] = None,
            name: str = "",
    ) -> ConvexTerm:
        """Add a nonlinear constraint to the model.

        Args:
            var: mip.Var or list of mip.Var or mip.LinExprTensor
            func: callable
            grad: callable
            name: str, default=''

        Returns: ConvexTerm
            The convex term representing the constraint.
        """
        term = ConvexTerm(
            var=var,
            func=func,
            grad=grad,
            step_size=self.step_size,
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
    ) -> ConvexTerm:
        """Add an objective term to the model.

        Args:
            var: mip.Var or list of mip.Var or mip.LinExprTensor
            func: callable
            grad: callable
            name: str, default=''

        Returns: ConvexTerm
            The objective term.
        """
        term = ConvexTerm(
            var=var,
            func=func,
            grad=grad,
            step_size=self.step_size,
            name=name,
        )
        self._objective_terms.append(term)
        return term

    def optimize(
            self,
            max_iters: int = 500,
            max_iters_no_improvement: Optional[int] = None,
            max_seconds_per_iter: Optional[float] = None,
    ) -> mip.OptimizationStatus:
        """Optimize the model.

        Args:
            max_iters: int, default=500
            max_iters_no_improvement: int or `None`, default=`None`
            max_seconds_per_iter:  int or `None`, default=`None`

        Returns: mip.OptimizationStatus
            The status of the search.
        """

        # Define objective in epigraph form
        bound = self._model.add_var(lb=-mip.INF, ub=mip.INF)
        self._model.objective = bound

        # Initialize search
        query_point = {x: self._start.get(x) or (x.lb + x.ub) / 2 for x in self._model.vars}
        iters_no_improvement = 0

        for i in range(max_iters):

            # Add cuts for violated nonlinear constraints
            for constr in self.nonlinear_constrs:
                if constr(query_point=query_point) > self.feasibility_tol:
                    expr = constr.generate_cut(query_point=query_point)
                    self._model.add_constr(expr <= 0)

            # Add objective cut
            expr = mip.xsum(term.generate_cut(query_point=query_point) for term in self.objective_terms)
            if self.minimize:
                self._model.add_constr(bound >= expr)
            else:
                self._model.add_constr(bound <= expr)

            # Re-optimize LP/MIP model
            status = self._model.optimize(max_seconds=max_seconds_per_iter or mip.INF)

            # If no solution is found, exit solve and return status
            if status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
                logging.info(f"Solve unsuccessful - exiting with optimization status: '{status.value}'.")
                return status

            # Update best solution/objective and query point
            solution = {var: var.x for var in self._model.vars}
            objective = sum(term(query_point=solution) for term in self.objective_terms)
            if (
                self.minimize == (objective < self.best_objective)
                and all(constr(solution) <= self.feasibility_tol for constr in self.nonlinear_constrs)
            ):
                iters_no_improvement = 0
                self._best_objective = objective
                self._best_solution = solution
            else:
                if np.isfinite(self.best_objective):
                    iters_no_improvement += 1
                if self.smoothing is not None:
                    query_point = {
                        var: self.smoothing * query_point[var] + (1 - self.smoothing) * solution[var]
                        for var in self._model.vars
                    }
                else:
                    query_point = query_point

            # Update best bound (clip values to prevent numerical errors from affecting termination logic)
            if self.minimize:
                self._best_bound = np.clip(bound.x, a_min=self.best_bound, a_max=self.best_objective)
            else:
                self._best_bound = np.clip(bound.x, a_min=self.best_objective, a_max=self.best_bound)

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
                if not i:
                    log_table_header(columns=self._search_log[-1].keys())
                if not i % self.log_freq:
                    log_table_row(values=self._search_log[-1].values())

            # Check early termination conditions
            if self.gap <= self.max_gap or self.gap_abs <= self.max_gap_abs:
                logging.info(f"Optimality tolerance reached - terminating search early.")
                return mip.OptimizationStatus.OPTIMAL
            if max_iters_no_improvement is not None:
                if iters_no_improvement > max_iters_no_improvement:
                    logging.info(
                        f"Max iterations without improvement reached - terminating search early.")
                return mip.OptimizationStatus.FEASIBLE

        logging.info(f"Max iterations reached - terminating search.")
        return mip.OptimizationStatus.FEASIBLE

    @property
    def start(self) -> Start:
        """Get the starting solution or partial solution provided."""
        return [(key, value) for key, value in self._start.items()]

    @start.setter
    def start(self, value: Start) -> None:
        """Set the starting solution or partial solution, provided as tuple of (variable, value) pairs."""
        # TODO add validation checks here
        self._start = {var: x for var, x in value}
        self._model.start = value

    @property
    def objective_terms(self) -> list[ConvexTerm]:
        """Get the objective terms of the model."""
        return self._objective_terms

    @property
    def nonlinear_constrs(self) -> list[ConvexTerm]:
        """Get the nonlinear constraints of the model."""
        return self._nonlinear_constraints

    @property
    def best_solution(self) -> dict[mip.Var, float]:
        """Get the best solution."""
        return self._best_solution

    @property
    def best_objective(self) -> float:
        """Get the best objective value."""
        return self._best_objective

    @property
    def best_bound(self) -> float:
        """Get the best bound."""
        return self._best_bound

    @property
    def gap(self) -> float:
        """Get the optimality gap."""
        return self.gap_abs / max(min(abs(self.best_objective), abs(self.best_bound)), 1e-10)

    @property
    def gap_abs(self) -> float:
        """Get the absolute optimality gap."""
        return abs(self.best_objective - self.best_bound)

    @property
    def search_log(self) -> pd.DataFrame:
        """Get the search log."""
        return pd.DataFrame(self._search_log).set_index("iteration")

    @staticmethod
    def sum(terms: Iterable[Union[mip.Var, mip.LinExpr]]) -> mip.LinExpr:
        """Create a linear expression from a summation."""
        return mip.xsum(terms=terms)

    def plot_search(
            self,
            log_scale: bool = False,
            show_gap: bool = False,
            max_iters: Optional[int] = None
    ) -> Figure:
        """Plot the search.

        Args:
            log_scale: bool, default=False
            show_gap: bool, default=False
            max_iters: int or `None`, default=`None`

        Returns: plotly Figure
        """
        df = self.search_log
        if max_iters is not None:
            df = df.iloc[:max_iters]
        return px.line(
            data_frame=df,
            y="gap" if show_gap else ["best_objective", "best_bound"],
            log_y=log_scale,
        )

    def _validate_params(self) -> None:
        check_scalar(
            x=self.max_gap,
            name="max_gap",
            lb=0.,
            var_type=float,
            include_boundaries=False,
        )
        check_scalar(
            x=self.max_gap_abs,
            name="max_gap_abs",
            lb=0.,
            var_type=float,
            include_boundaries=False,
        )
        check_scalar(
            x=self.feasibility_tol,
            name="feasibility_tol",
            var_type=float,
            lb=0.,
            include_boundaries=False,
        )
        if self.smoothing is not None:
            check_scalar(
                x=self.smoothing,
                name="smoothing",
                var_type=float,
                lb=0.,
                ub=1.,
                include_boundaries=False,
            )
        if self.log_freq is not None:
            check_scalar(
                x=self.log_freq,
                name="log_freq",
                var_type=int,
                lb=0,
                include_boundaries=False,
            )

    @staticmethod
    def _validate_bounds(lb: float, ub: float, var_type: str) -> tuple[float, float]:
        if var_type == mip.BINARY:
            lb, ub = 0, 1
        else:
            check_scalar(
                x=lb,
                name="lb",
                var_type=(float, int),
                ub=ub,
                lb=-mip.INF,
                include_boundaries=False,
            )
            check_scalar(
                x=ub,
                name="ub",
                var_type=(float, int),
                ub=mip.INF, lb=lb,
                include_boundaries=False,
            )
        return lb, ub


