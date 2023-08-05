import logging
import numbers
from time import time
from typing import Optional, Union
import mip

from .objective_term import ObjectiveTerm, Variables


class Model(mip.Model):

    def __init__(
            self,
            name: str = "",
            sense: str = mip.MINIMIZE,
            solver_name: str = "",
            solver: Optional[mip.Solver] = None,
            max_gap: float = 1e-6,
            verbose: bool = False,
    ):
        super().__init__(
            name=name,
            sense=sense,
            solver_name=solver_name,
            solver=solver,
        )
        self.max_gap = max_gap
        self.verbose = verbose
        self._objective_terms: list[ObjectiveTerm] = list()
        self._initialization_objective: Optional[mip.LinExpr] = None
        self._search_log: list[dict[str, numbers.Real]] = list()

    def add_objective_term(
            self,
            variables: Variables,
            fun,
            grad=None
    ) -> ObjectiveTerm:
        objective_term = ObjectiveTerm(
            var=variables,
            fun=fun,
            grad=grad,
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
        if self.verbose:
            logging.info("Performing initialization solve...")
        status = super().optimize(
            max_seconds=max_seconds,
            max_nodes=max_nodes,
            max_solutions=max_solutions,
            max_seconds_same_incumbent=max_seconds_same_incumbent,
            max_nodes_same_incumbent=max_nodes_same_incumbent,
        )
        self.reset()

        # If no feasible solution found, exit solve and return status
        if status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
            if self.verbose:
                logging.info(f"Initial solve unsuccessful - exiting with optimization status: '{status.value}'")
            return status

        # Define objective in epigraph form
        objective = self.add_var(lb=-mip.INF)
        self.objective = objective

        for i in range(max_iters):

            # Set starting solution for discrete variables
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
                if self.verbose:
                    logging.info(f"Solve unsuccessful - exiting with optimization status: '{status.value}'.")
                return status

            # Update
            incumbent, bound = self.objective_value, super().objective_value
            gap = abs(incumbent - bound) / min(abs(incumbent), abs(bound))
            self._search_log.append(
                {
                    "iteration": i,
                    "time_elapsed": time() - start_time,
                    "incumbent": incumbent,
                    "bound": bound
                }
            )

            # Check early termination condition
            if gap <= self.max_gap:
                if self.verbose:
                    logging.info(f"Optimality tolerance reached - terminating search early.")
                return mip.OptimizationStatus.OPTIMAL

        if self.verbose:
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
