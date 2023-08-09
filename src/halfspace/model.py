import logging
import numbers
from time import time
from typing import Optional, Union

import mip
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from .objective_term import ObjectiveTerm, Variables, Fun, Grad
from .utils import log_table_header, log_table_row


logging.getLogger("mip").setLevel(logging.WARNING)


class Model(mip.Model):

    def __init__(
            self,
            name: str = "",
            sense: str = mip.MINIMIZE,
            solver_name: str = "",
            solver: Optional[mip.Solver] = None,
            max_gap: float = 1e-6,
            step_size: float = 1e-6,
    ):
        """Construct for optimization model.

        Args:
            name: str, default=""
            sense: str, default=mip.MINIMIZE
            solver_name: str, default=""
            solver: mip.Solver, default=None
            max_gap: float, default=1e-6
            step_size: float, default=1e-6
        """
        super().__init__(
            name=name,
            sense=sense,
            solver_name=solver_name,
            solver=solver,
        )
        self.max_gap = max_gap
        self.step_size = step_size
        self._objective_terms: list[ObjectiveTerm] = list()
        self._initialization_objective: Optional[mip.LinExpr] = None
        self._search_log: list[dict[str, numbers.Real]] = list()

    def add_objective_term(
            self,
            var: Variables,
            func: Fun,
            grad: Optional[Grad] = None,
    ) -> ObjectiveTerm:
        """

        Args:
            var:
            func:
            grad:

        Returns:

        """
        objective_term = ObjectiveTerm(
            var=var,
            func=func,
            grad=grad,
            step_size=self.step_size,
        )
        self._objective_terms.append(objective_term)
        return objective_term

    def optimize(
        self,
        max_iters: int = 100,
        max_seconds: numbers.Real = mip.INF,
        max_nodes: int = mip.INT_MAX,
        max_solutions: int = mip.INT_MAX,
        max_seconds_same_incumbent: numbers.Real = mip.INF,
        max_nodes_same_incumbent: int = mip.INT_MAX,
        relax: bool = False,
    ) -> mip.OptimizationStatus:
        """

        Args:
            max_iters:
            max_seconds:
            max_nodes:
            max_solutions:
            max_seconds_same_incumbent:
            max_nodes_same_incumbent:
            relax:

        Returns: mip.OptimizationStatus

        """

        # Check that at least one objective term has been added
        assert self.objective_terms, (
            "No objective terms have been added to the model"
            " - at least one objective term must be added prior to optimization."
        )

        # Get start time for progress log
        start_time = time()

        # Find an initial feasible solution
        if self.initialization_objective is None:
            self.objective = 1.
        else:
            self.objective = self.initialization_objective
        logging.info("Performing initialization solve...")
        status = super().optimize(
            max_seconds=max_seconds,
            max_nodes=max_nodes,
            max_solutions=max_solutions,
            max_seconds_same_incumbent=max_seconds_same_incumbent,
            max_nodes_same_incumbent=max_nodes_same_incumbent,
        )

        # If no feasible solution found, exit solve and return status
        if status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
            logging.info(f"Initial solve unsuccessful - exiting with optimization status: '{status.value}'")
            return status

        # Define objective in epigraph form
        objective = self.add_var(lb=-mip.INF)
        self.objective = objective

        for i in range(max_iters):

            # Set starting solution for discrete variables
            # Note: the attribute `vars` of the Model class from the `mip` package accidentally shadows a built-in name
            self.start = [(var, var.x) for var in self.vars if var.var_type in (mip.BINARY, mip.INTEGER)]

            # Add new cut
            expr = mip.xsum(term.generate_cut() for term in self._objective_terms)
            if self.sense == mip.MINIMIZE:
                self.add_constr(objective >= expr)
            else:
                self.add_constr(objective <= expr)

            # Re-optimize model
            status = super().optimize(
                max_seconds=max_seconds,
                max_nodes=max_nodes,
                max_solutions=max_solutions,
                max_seconds_same_incumbent=max_seconds_same_incumbent,
                max_nodes_same_incumbent=max_nodes_same_incumbent,
            )

            # If no feasible solution found, exit solve and return status
            if status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
                logging.info(f"Solve unsuccessful - exiting with optimization status: '{status.value}'.")
                return status

            # Update search log
            incumbent, bound = self.objective_value, super().objective_value
            gap = abs(incumbent - bound) / max(min(abs(incumbent), abs(bound)), self.max_gap ** 2)
            self._search_log.append(
                {
                    "iteration": i,
                    "time_elapsed": time() - start_time,
                    "incumbent": incumbent,
                    "bound": bound,
                    "gap": gap,
                }
            )
            row = self._search_log[-1]
            if not i:
                log_table_header(columns=row.keys())
            log_table_row(values=row.values())
            # print({key: "{0:.3e}".format(value) if isinstance(value, float) else value for key, value in row.items()})

            # Check early termination condition
            if gap <= self.max_gap:
                logging.info(f"Optimality tolerance reached - terminating search early.")
                return mip.OptimizationStatus.OPTIMAL

        logging.info(f"Max iterations reached - terminating search.")
        return mip.OptimizationStatus.FEASIBLE

    def clear(self) -> None:
        super().clear()
        self._objective_terms = list()
        self._initialization_objective = None

    @property
    def objective_terms(self) -> list[ObjectiveTerm]:
        return self._objective_terms

    @property
    def objective_value(self) -> float:
        return sum(term.value for term in self.objective_terms)

    @property
    def initialization_objective(self) -> Optional[mip.LinExpr]:
        return self._initialization_objective

    @initialization_objective.setter
    def initialization_objective(self, expr: Union[mip.LinExpr, mip.Var]):
        if not isinstance(expr, (mip.LinExpr, mip.Var)):
            raise TypeError
        self._initialization_objective = expr

    @property
    def search_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._search_log)

    def plot_search(
            self,
            log_scale: bool = False,
            x: str = "iteration",
            gap: bool = False
    ) -> Figure:
        return px.line(
            data_frame=self.search_log,
            x=x,
            y="gap" if gap else ["incumbent", "bound"],
            log_y=log_scale,
        )
