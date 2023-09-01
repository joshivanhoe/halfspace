import logging
from typing import Optional, Iterable, Union

import mip
import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
import plotly.express as px

from .objective_term import ObjectiveTerm, Variables, Fun, Grad
from .utils import _log_table_header, _log_table_row, _sigmoid
from time import time

Start = list[tuple[mip.Var, float]]


class Model:

    def __init__(
            self,
            minimize: bool = True,
            max_gap: float = 1e-4,
            max_mip_gap: float = 1e-6,
            min_update_weight: float = 0.1,
            update_smoothing: float = 1.,
            solver_name: Optional[str] = None,
    ):
        self.minimize = minimize
        self.max_gap = max_gap
        self.max_mip_gap = max_mip_gap
        self.min_update_weight = min_update_weight
        self.update_smoothing = update_smoothing,
        self.solver_name = solver_name
        self.reset()

    def reset(self) -> None:
        """Reset the model."""
        self._mip_model: mip.Model = mip.Model(
            solver_name=self.solver_name,
            sense=mip.MINIMIZE if self.minimize else mip.MAX,
        )
        self._mip_model.verbose = 0
        self._mip_model.max_mip_gap = self.max_mip_gap
        self._start: dict[int, tuple[mip.Var, float]] = dict()
        self._objective_terms: list[ObjectiveTerm] = list()
        self._search_log: list[dict[str, float]] = list()

    def add_variable(
            self,
            lb: float = 0.,
            ub: float = mip.INF,
            var_type: str = mip.CONTINUOUS,
            name: str = ""
    ) -> mip.Var:
        """Add a decision variable to the model.

        Args:
            lb: float, default=0.
            ub: float, default=inf
            var_type: str, default='C'
            name: str, default=''
                The name of the decision variable

        Returns: mip.Var
            The decision variable.
        """
        return self._mip_model.add_var(lb=lb, ub=ub, var_type=var_type, name=name)

    def add_variable_tensor(
            self,
            shape: tuple[int, ...],
            lb: float = 0,
            ub: float = mip.INF,
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
        return self._mip_model.add_var_tensor(
            shape=shape,
            lb=lb,
            ub=ub,
            var_type=var_type,
            name=name,
        )

    def add_constraint(self, constraint: mip.LinExpr, name: str = "") -> mip.Constr:
        """Add a linear constraint to the model.

        Args:
            constraint: mip.LinExpr
            name: str, default=''
                The name of the constraint.

        Returns: mip.Constr
            The constraint
        """
        return self._mip_model.add_constr(lin_expr=constraint, name=name)

    def add_objective_term(
            self,
            var: Variables,
            func: Fun,
            grad: Optional[Grad] = None,
            name: str = "",
            step_size: float = 1e-6,
    ) -> ObjectiveTerm:
        """Add an objective term to the model.

        Args:
            var: mip.Var or list of mip.Var or mip.LinExprTensor
            func: callable
            grad: callable
            name: str, default=''
            step_size: float, default=1e-6

        Returns: ObjectiveTerm
            The objective term
        """
        objective_term = ObjectiveTerm(
            var=var,
            func=func,
            grad=grad,
            step_size=step_size,
            name=name,
        )
        self._objective_terms.append(objective_term)
        return objective_term

    def optimize(
            self,
            max_iters: int = 100,
            max_iters_no_improvement: Optional[int] = None,
            max_seconds_per_cut: Optional[float] = None,
    ) -> mip.OptimizationStatus:

        start_time = time()

        # Define objective in epigraph form
        objective = self.add_variable(lb=-mip.INF, name="_objective")
        self._mip_model.objective = objective

        # Initialize query point
        query_point = list()
        for term in self.objective_terms:
            if term.is_multivariable:
                query_point.append(np.array([self._default_value(x=x) for x in term.var]))
            else:
                query_point.append(self._default_value(x=term.var))

        # Initialize search variables
        if self.minimize:
            best = mip.INF
        else:
            best = -mip.INF
        incumbent = best
        gap = mip.INF
        n_iters_no_improvement = 0

        for i in range(max_iters):
            if i:
                # Update query point
                if n_iters_no_improvement:
                    update_weight = max(
                        _sigmoid(-abs(incumbent - best) / max(min(abs(incumbent), abs(best)), 1e-10), scale=100),
                        self.min_update_weight
                    )

                else:
                    update_weight = 1.
                query_point = [x + update_weight * (term.x - x) for term, x in zip(self.objective_terms, query_point)]

                # Update MIP warm start
                self._mip_model.start = [
                    (var, var.x) for var in self._mip_model.vars
                    if var.var_type in (mip.BINARY, mip.INTEGER)
                ]

            # Add cutting plane
            expr = mip.xsum(term.generate_cut(x=x) for term, x in zip(self.objective_terms, query_point))
            if self.minimize:
                self.add_constraint(objective >= expr, name=f"_cut_{i}")
            else:
                self.add_constraint(objective <= expr, name=f"_cut_{i}")

            # Re-optimize MIP model
            status = self._mip_model.optimize(max_seconds=max_seconds_per_cut)

            # If no feasible solution found, exit solve and return status
            if status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
                logging.info(f"Solve unsuccessful - exiting with optimization status: '{status.value}'.")
                return status

            # Update search log
            incumbent, bound = self.objective_value, float(objective.x)
            if (self.minimize and incumbent < best) or (not self.minimize and incumbent > best):
                n_iters_no_improvement = 0
                best = incumbent
                gap = abs(best - bound) / max(min(abs(best), abs(bound)), self.max_gap ** 2)
            else:
                n_iters_no_improvement += 1
            row = {
                "iteration": i,
                "time": time() - start_time,
                "incumbent": incumbent,
                "best": best,
                "bound": bound,
                "gap": gap,
            }
            self._search_log.append(row)
            if not i:
                _log_table_header(columns=row.keys())
            _log_table_row(values=row.values())

            # Check early termination conditions
            if gap <= self.max_gap:
                logging.info(f"Optimality tolerance reached - terminating search early.")
                return mip.OptimizationStatus.OPTIMAL
            if n_iters_no_improvement > max_iters_no_improvement:
                logging.info(f"Max iterations without improvement reached - terminating search early.")
                return mip.OptimizationStatus.FEASIBLE

        logging.info(f"Max iterations reached - terminating search.")
        return mip.OptimizationStatus.FEASIBLE

    def _default_value(self, x: mip.Var) -> float:
        start = self._start.get(hash(x))
        if start:
            return start[1]
        lb_finite = np.isfinite(x.lb)
        ub_finite = np.isfinite(x.ub)
        if lb_finite and ub_finite:
            return (x.lb + x.ub) / 2
        if lb_finite:
            return float(x.lb)
        if ub_finite:
            return float(x.ub)
        return 0.

    @property
    def start(self) -> Start:
        return list(self._start.values())

    @start.setter
    def start(self, value: Start) -> None:
        # TODO add validation checks here
        self._start = {hash(var): (var, x) for var, x in value}
        self._mip_model.start = value

    @property
    def objective_terms(self) -> list[ObjectiveTerm]:
        return self._objective_terms

    @property
    def objective_value(self) -> float:
        return sum(term.value for term in self.objective_terms)

    @property
    def search_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._search_log)

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
