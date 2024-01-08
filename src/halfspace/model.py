import logging
from typing import Optional, Iterable, Union

import mip
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from .search_state import SearchState
from .term import NonlinearTerm, Var, Fun, Grad
from .utils import check_scalar

Start = list[tuple[mip.Var, float]]


class Model:

    def __init__(
            self,
            minimize: bool = True,
            max_gap: float = 1e-4,
            max_gap_abs: float = 1e-4,
            solver_name: Optional[str] = None,
            log_freq: Optional[int] = 1,
    ):
        """

        Args:
            minimize: bool, default=`True`
            max_gap: float, default=1e-4
            max_gap_abs: float, default=1e-4
            solver_name: str or `None`, default=`None`
            log_freq: int or `None`, default = 1
        """
        self.minimize = minimize
        self.max_gap = max_gap
        self.max_gap_abs = max_gap_abs
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
        self._start: dict[mip.Var, float] = dict()
        self._objective_terms: list[NonlinearTerm] = list()
        self._nonlinear_constraints: list[NonlinearTerm] = list()
        self._search_state: Optional[SearchState] = None

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
        if var_type == mip.BINARY:
            lb, ub = 0, 1
        else:
            check_scalar(x=lb, name="lb", var_type=(float, int), ub=ub, lb=-mip.INF, include_boundaries=False)
            check_scalar(x=ub, name="ub", var_type=(float, int), ub=mip.INF, lb=lb, include_boundaries=False)
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
        if var_type == mip.BINARY:
            lb, ub = 0, 1
        else:
            check_scalar(x=lb, name="lb", var_type=(float, int), ub=ub, lb=-mip.INF, include_boundaries=False)
            check_scalar(x=ub, name="ub", var_type=(float, int), ub=mip.INF, lb=lb, include_boundaries=False)
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
            step_size: float = 1e-6,
    ) -> NonlinearTerm:
        """Add a nonlinear constraint to the model.

        Args:
            var: mip.Var or list of mip.Var or mip.LinExprTensor
            func: callable
            grad: callable
            name: str, default=''
            step_size: float, default=1e-6

        Returns: NonlinearTerm
            The constraint
        """
        term = NonlinearTerm(
            var=var,
            func=func,
            grad=grad,
            step_size=step_size,
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
            step_size: float = 1e-6,
    ) -> NonlinearTerm:
        """Add an objective term to the model.

        Args:
            var: mip.Var or list of mip.Var or mip.LinExprTensor
            func: callable
            grad: callable
            name: str, default=''
            step_size: float, default=1e-6

        Returns: NonlinearTerm
            The objective term
        """
        term = NonlinearTerm(
            var=var,
            func=func,
            grad=grad,
            step_size=step_size,
            name=name,
        )
        self._objective_terms.append(term)
        return term

    def optimize(
            self,
            max_iters: int = 100,
            max_iters_without_improvement: Optional[int] = None,
            max_seconds_per_cut: Optional[float] = None,
    ) -> mip.OptimizationStatus:

        # Define objective in epigraph form
        objective = self.add_var(lb=-mip.INF, ub=mip.INF, name="_objective")
        self._mip_model.objective = objective

        # Initialize query point and search state
        query_point = {x: self._start.get(x) or (x.lb + x.ub) / 2 for x in self._mip_model.vars}
        self._search_state = SearchState(minimize=self.minimize)

        for i in range(max_iters):

            # Add objective cut
            expr = mip.xsum(term.generate_cut(query_point=query_point) for term in self.objective_terms)
            if self.minimize:
                self.add_linear_constraint(objective >= expr)
            else:
                self.add_linear_constraint(objective <= expr)

            # Re-optimize MIP model
            status = self._mip_model.optimize(max_seconds=max_seconds_per_cut)

            # If no feasible solution found, exit solve and return status
            if status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
                logging.info(f"Solve unsuccessful - exiting with optimization status: '{status.value}'.")
                return status

            # Update query point
            query_point = {var: var.x for var in self._mip_model.vars}

            # Add cuts for violated nonlinear constraints
            is_feasible = True
            for constraint in self.nonlinear_constraints:
                expr = constraint.generate_cut(query_point=query_point)
                if expr is not None:  # If the constraint is not violated, the cut expression will be `None`
                    is_feasible = False
                    self.add_linear_constraint(expr <= 0)

            # Update search state
            if is_feasible:
                incumbent, bound = self.objective_value, float(objective.x)
            else:
                incumbent, bound = None, None
            self._search_state.update(incumbent=incumbent, bound=bound)

            # Check early termination conditions
            if self._search_state.gap <= self.max_gap or self._search_state.gap_abs <= self.max_gap_abs:
                logging.info(f"Optimality tolerance reached - terminating search early.")
                return mip.OptimizationStatus.OPTIMAL
            if max_iters_without_improvement is not None:
                if self._search_state.iterations_without_improvement > max_iters_without_improvement:
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
    def nonlinear_constraints(self) -> list[NonlinearTerm]:
        return self._nonlinear_constraints

    @property
    def objective_terms(self) -> list[NonlinearTerm]:
        return self._objective_terms

    @property
    def objective_value(self) -> float:
        return sum(term.value for term in self.objective_terms)

    @property
    def search_log(self) -> pd.DataFrame:
        return self._search_state.log

    @staticmethod
    def sum(terms: Iterable[Union[mip.Var, mip.LinExpr]]) -> mip.LinExpr:
        return mip.xsum(terms=terms)

    def plot_search(
            self,
            log_scale: bool = False,
            time_for_x: bool = False,
            gap_for_y: bool = False,
            max_iters: Optional[int] = None
    ) -> Figure:
        df = self.search_log
        if max_iters is not None:
            df = df.iloc[:max_iters]
        return px.line(
            data_frame=df,
            x="time" if time_for_x else "iteration",
            y="gap" if gap_for_y else ["best", "bound"],
            log_y=log_scale,
        )
