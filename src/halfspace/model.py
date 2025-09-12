"""This module implements the Model class.

It provides users with a general purpose API for modelling and solving mixed-integer convex optimization problems.
"""

import logging
from typing import Iterable

import mip
import numpy as np
import pandas as pd

from .convex_term import ConvexTerm, Var, Func, FuncGrad, Grad
from .utils import check_scalar, log_table_header, log_table_row

type Start = list[tuple[mip.Var, float]]


class Model:
    """Mixed-integer convex optimization model using outer approximation.
    
    This class implements an outer approximation algorithm for solving mixed-integer convex
    optimization problems. The algorithm iteratively adds linear cuts to approximate nonlinear
    constraints and objective functions, solving a sequence of mixed-integer linear programs.
    
    The model supports both continuous and discrete variables, linear and nonlinear constraints,
    and can handle both minimization and maximization problems (with concave objectives for
    maximization).

    Attributes:
        minimize: Whether the objective should be minimized. If `False`, the objective will be
            maximized - note that in this case the objective must be concave, not convex.
        max_gap: The maximum relative optimality gap allowed before the search is terminated.
        max_gap_abs: The maximum absolute optimality gap allowed before the search is terminated.
        infeasibility_tol: The maximum allowed constraint violation permitted for a solution to
            be considered feasible.
        step_size: The step size used to numerically evaluate gradients using the central finite
            difference method. Only used when a function for analytically computing the gradient
            is not provided.
        smoothing: The smoothing parameter used to update the query point. If `None`, the query
            point will be updated to the incumbent solution at each iteration.
        solver_name: The MIP solver to use. Valid options are 'CBC' and 'GRB' (Gurobi). Note that
            'GRB' requires a license.
        log_freq: The frequency with which progress logs are printed during optimization.
            If `None`, no progress logs are printed.
    """

    def __init__(
        self,
        minimize: bool = True,
        max_gap: float = 1e-4,
        max_gap_abs: float = 1e-4,
        infeasibility_tol: float = 1e-4,
        step_size: float = 1e-6,
        smoothing: float | None = 0.5,
        solver_name: str | None = "CBC",
        log_freq: int | None = 1,
    ) -> None:
        """Initialize the optimization model.

        Args:
            minimize: Whether to minimize the objective. If `False`, maximizes (requires concave objective).
            max_gap: Maximum relative optimality gap for early termination. Must be positive.
            max_gap_abs: Maximum absolute optimality gap for early termination. Must be positive.
            infeasibility_tol: Maximum constraint violation for feasible solutions. Must be positive.
            step_size: Step size for numerical gradient approximation. Must be positive.
            smoothing: Query point smoothing parameter in (0, 1). If `None`, uses incumbent solution.
            solver_name: MIP solver name ('CBC' or 'GRB'). 'GRB' requires a license.
            log_freq: Progress logging frequency. If `None`, no progress logs are printed.
        """
        self.minimize = minimize
        self.max_gap = max_gap
        self.max_gap_abs = max_gap_abs
        self.infeasibility_tol = infeasibility_tol
        self.step_size = step_size
        self.smoothing = smoothing
        self.solver_name = solver_name
        self.log_freq = log_freq
        self._validate_params()
        self.reset()

    def reset(self) -> None:
        """Reset the model to its initial state.
        
        Clears all variables, constraints, and solution data, returning the model to
        the state it was in immediately after construction.
        """
        self._model: mip.Model = mip.Model(
            solver_name=self.solver_name,
            sense=mip.MINIMIZE if self.minimize else mip.MAXIMIZE,
        )
        self._model.verbose = 0
        self._model.infeas_tol = self.infeasibility_tol
        self._start: dict[mip.Var, float] = dict()
        self._objective_terms: list[ConvexTerm] = list()
        self._nonlinear_constrs: list[ConvexTerm] = list()
        self._best_solution: dict[mip.Var, float] = dict()
        self._objective_value: float = (1 if self.minimize else -1) * mip.INF
        self._best_bound: float = -self._objective_value
        self._status: mip.OptimizationStatus | None = None
        self._search_log: list[dict[str, float]] = list()

    def add_var(
        self,
        lb: float | int = 0,
        ub: float | int = mip.INF,
        var_type: str = mip.CONTINUOUS,
        name: str = "",
    ) -> mip.Var:
        """Add a single decision variable to the model.

        Args:
            lb: Lower bound for the variable. Must be finite and less than upper bound.
                Cannot be `None` if `var_type` is 'C' or 'I'.
            ub: Upper bound for the variable. Must be finite and greater than lower bound.
                Cannot be `None` if `var_type` is 'C' or 'I'.
            var_type: Variable type. Valid options are 'C' (continuous), 'I' (integer), and 'B' (binary).
            name: Optional name for the variable.

        Returns:
            The created decision variable.
        """
        lb, ub = self._validate_bounds(lb=lb, ub=ub, var_type=var_type)
        return self._model.add_var(lb=lb, ub=ub, var_type=var_type, name=name)

    def add_var_tensor(
        self,
        shape: tuple[int, ...],
        lb: float | int = 0,
        ub: float | int = mip.INF,
        var_type: str = mip.CONTINUOUS,
        name: str = "",
    ) -> mip.LinExprTensor:
        """Add a tensor of decision variables to the model.

        Args:
            shape: Shape of the variable tensor.
            lb: Lower bound for all variables. Must be finite and less than upper bound.
                Cannot be `None` if `var_type` is 'C' or 'I'.
            ub: Upper bound for all variables. Must be finite and greater than lower bound.
                Cannot be `None` if `var_type` is 'C' or 'I'.
            var_type: Variable type for all variables. Valid options are 'C' (continuous), 'I' (integer), and 'B' (binary).
            name: Base name for the variables (indices will be appended).

        Returns:
            The created variable tensor.
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
            constraint: Linear constraint expression (e.g., x + y <= 1).
            name: Optional name for the constraint.

        Returns:
            The created constraint object.
        """
        return self._model.add_constr(lin_expr=constraint, name=name)

    def add_nonlinear_constr(
        self,
        var: Var,
        func: Func | FuncGrad,
        grad: Grad | bool | None = None,
        name: str = "",
    ) -> ConvexTerm:
        """Add a nonlinear constraint to the model.

        The constraint is enforced as func(var) <= 0. The function must be convex for
        minimization problems or concave for maximization problems.

        Args:
            var: Variable(s) in the constraint. Can be a single variable, iterable of variables,
                or variable tensor.
            func: Function computing the constraint value. Should accept one argument for each
                variable in `var`. If `var` is a tensor, function should accept a single array.
            grad: Function computing the gradient. Should accept same arguments as `func`.
                If `None`, gradient is approximated numerically. If `True`, `func` should return
                (value, gradient) tuple for efficiency.
            name: Optional name for the constraint.

        Returns:
            The convex term representing the constraint.
        """
        term = ConvexTerm(
            var=var,
            func=func,
            grad=grad,
            step_size=self.step_size,
            name=name,
        )
        self._nonlinear_constrs.append(term)
        return term

    def add_objective_term(
        self,
        var: Var,
        func: Func | FuncGrad,
        grad: Grad | bool | None = None,
        name: str = "",
    ) -> ConvexTerm:
        """Add a term to the objective function.

        The function must be convex for minimization problems or concave for maximization
        problems. Multiple terms can be added to build up a complex objective.

        Args:
            var: Variable(s) in the objective term. Can be a single variable, iterable of
                variables, or variable tensor.
            func: Function computing the objective value. Should accept one argument for each
                variable in `var`. If `var` is a tensor, function should accept a single array.
            grad: Function computing the gradient. Should accept same arguments as `func`.
                If `None`, gradient is approximated numerically. If `True`, `func` should return
                (value, gradient) tuple for efficiency.
            name: Optional name for the objective term.

        Returns:
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
        max_iters: int = 100,
        max_iters_no_improvement: int | None = None,
        max_seconds_per_iter: float | None = None,
    ) -> mip.OptimizationStatus:
        """Solve the optimization problem using outer approximation.

        The algorithm iteratively adds linear cuts to approximate nonlinear constraints
        and objective functions, solving a sequence of mixed-integer linear programs.

        Args:
            max_iters: Maximum number of outer approximation iterations.
            max_iters_no_improvement: Maximum iterations without objective improvement after
                finding a feasible solution. If `None`, continues until `max_iters`.
            max_seconds_per_iter: Maximum seconds for the MIP solver per iteration.
                If `None`, solver runs until convergence.

        Returns:
            Optimization status indicating success or failure.
        """
        # Set up epigraph formulation: minimize/maximize t subject to t >=/<= objective
        bound = self._model.add_var(lb=-mip.INF, ub=mip.INF)
        self._model.objective = bound

        # Initialize search with starting point or variable bounds midpoint
        query_point = {x: self._start.get(x) or (x.lb + x.ub) / 2 for x in self._model.vars}
        iters_no_improvement = 0

        for i in range(max_iters):
            # Add linear cuts for violated nonlinear constraints
            for constr in self.nonlinear_constrs:
                if constr(query_point=query_point) > self.infeasibility_tol:
                    expr = constr.generate_cut(query_point=query_point)
                    self._model.add_constr(expr <= 0)

            # Add linear cut for objective function
            expr = mip.xsum(term.generate_cut(query_point=query_point) for term in self.objective_terms)
            if self.minimize:
                self._model.add_constr(bound >= expr)  # t >= objective
            else:
                self._model.add_constr(bound <= expr)  # t <= objective

            # Solve the current mixed-integer linear program
            status = self._model.optimize(max_seconds=max_seconds_per_iter or mip.INF)

            # Check if solver found a feasible solution
            if status not in (
                mip.OptimizationStatus.OPTIMAL,
                mip.OptimizationStatus.FEASIBLE,
            ):
                logging.info(f"Solve unsuccessful - exiting with optimization status: '{status.value}'.")
                self._status = status
                return self.status

            # Extract solution and evaluate true objective value
            solution = {var: var.x for var in self._model.vars}
            objective_value_new = sum(term(query_point=solution) for term in self.objective_terms)
            
            # Check if this is a better feasible solution
            is_improvement = self.minimize == (objective_value_new < self.objective_value)
            is_feasible = all(constr(solution) <= self.infeasibility_tol for constr in self.nonlinear_constrs)
            
            if is_improvement and is_feasible:
                iters_no_improvement = 0
                self._objective_value = objective_value_new
                self._best_solution = solution
            else:
                if np.isfinite(self.objective_value):
                    iters_no_improvement += 1
                
                # Update query point for next iteration
                if self.smoothing is not None:
                    # Smooth between current query point and new solution
                    query_point = {
                        var: self.smoothing * query_point[var] + (1 - self.smoothing) * solution[var]
                        for var in self._model.vars
                    }
                else:
                    # Use incumbent solution as next query point
                    query_point = solution

            # Update best bound with monotonicity to prevent numerical issues
            if self.minimize:
                self._best_bound = np.clip(bound.x, a_min=self.best_bound, a_max=self.objective_value)
            else:
                self._best_bound = np.clip(bound.x, a_min=self.objective_value, a_max=self.best_bound)

            # Log progress
            self._search_log.append(
                {
                    "iteration": i,
                    "objective_value": self.objective_value,
                    "best_bound": self.best_bound,
                    "gap": self.gap,
                }
            )
            if self.log_freq is not None:
                if not i:
                    log_table_header(columns=self._search_log[-1].keys())
                if not i % self.log_freq:
                    log_table_row(values=self._search_log[-1].values())

            # Check convergence criteria
            if self.gap <= self.max_gap or self.gap_abs <= self.max_gap_abs:
                logging.info("Optimality tolerance reached - terminating search early.")
                self._status = mip.OptimizationStatus.OPTIMAL
                return self.status
            if max_iters_no_improvement is not None:
                if iters_no_improvement >= max_iters_no_improvement:
                    logging.info("Max iterations without improvement reached - terminating search early.")
                    self._status = mip.OptimizationStatus.FEASIBLE
                    return self.status

        # Reached maximum iterations
        logging.info("Max iterations reached - terminating search.")
        if self.best_solution:
            self._status = mip.OptimizationStatus.FEASIBLE
        else:
            self._status = mip.OptimizationStatus.NO_SOLUTION_FOUND
        return self.status

    def var_by_name(self, name: str) -> mip.Var:
        """Get a variable by its name.
        
        Args:
            name: The name of the variable to retrieve.
            
        Returns:
            The variable with the specified name.
            
        Raises:
            KeyError: If no variable with the given name exists.
        """
        return self._model.var_by_name(name=name)

    def var_value(self, x: mip.Var | mip.LinExprTensor | str) -> float | np.ndarray:
        """Get the value of one or more variables from the best solution.

        Args:
            x: Variable(s) to get values for. Can be a single variable, variable tensor,
                variable name (string), or iterable of variables.

        Returns:
            Variable value(s) as float or numpy array.
            
        Raises:
            TypeError: If input type is not supported.
        """
        if isinstance(x, str):
            x = self.var_by_name(name=x)
        if isinstance(x, mip.Var):
            return self.best_solution[x]
        if isinstance(x, mip.LinExprTensor):
            return np.array([self.best_solution[var] for var in x.flatten()]).reshape(x.shape)
        if isinstance(x, Iterable):
            return np.array([self.best_solution[var] for var in x])
        raise TypeError(f"Input of type '{type(x)}' not supported.")

    @property
    def objective_terms(self) -> list[ConvexTerm]:
        """Get the objective terms of the model.
        
        Returns:
            List of convex terms that make up the objective function.
        """
        return self._objective_terms

    @property
    def linear_constrs(self) -> mip.ConstrList:
        """Get the linear constraints of the model.

        After optimization, this includes both original linear constraints and
        the linear cuts added during the outer approximation process.
        
        Returns:
            List of all linear constraints in the model.
        """
        return self._model.constrs

    @property
    def nonlinear_constrs(self) -> list[ConvexTerm]:
        """Get the nonlinear constraints of the model.
        
        Returns:
            List of convex terms representing nonlinear constraints.
        """
        return self._nonlinear_constrs

    @property
    def start(self) -> Start:
        """Get the starting solution or partial solution.
        
        Returns:
            List of (variable, value) pairs defining the starting point.
        """
        return [(key, value) for key, value in self._start.items()]

    @start.setter
    def start(self, value: Start) -> None:
        """Set the starting solution or partial solution.
        
        Args:
            value: List of (variable, value) pairs defining the starting point.
        """
        # TODO add validation checks here
        self._start = {var: x for var, x in value}
        self._model.start = value

    @property
    def best_solution(self) -> dict[mip.Var, float]:
        """Get the best feasible solution found.
        
        Returns:
            Dictionary mapping variables to their values in the best solution.
        """
        return self._best_solution

    @property
    def objective_value(self) -> float:
        """Get the objective value of the best solution.
        
        Returns:
            Objective function value at the best feasible solution.
        """
        return self._objective_value

    @property
    def best_bound(self) -> float:
        """Get the best bound on the optimal objective value.
        
        Returns:
            Best known bound (lower bound for minimization, upper bound for maximization).
        """
        return self._best_bound

    @property
    def gap(self) -> float:
        """Get the relative optimality gap.
        
        Returns:
            Relative gap as |objective_value - best_bound| / max(|objective_value|, |best_bound|).
        """
        return self.gap_abs / max(abs(self.objective_value), abs(self.best_bound), 1.0)

    @property
    def gap_abs(self) -> float:
        """Get the absolute optimality gap.
        
        Returns:
            Absolute gap as |objective_value - best_bound|.
        """
        return abs(self.objective_value - self.best_bound)

    @property
    def status(self) -> mip.OptimizationStatus:
        """Get the optimization status.
        
        Returns:
            Status indicating whether optimization was successful and why it terminated.
        """
        return self._status

    @property
    def search_log(self) -> pd.DataFrame:
        """Get the search progress log.
        
        Returns:
            DataFrame with columns: iteration, objective_value, best_bound, gap.
        """
        return pd.DataFrame(self._search_log).set_index("iteration")

    @staticmethod
    def sum(terms: Iterable[mip.Var | mip.LinExpr]) -> mip.LinExpr:
        """Create a linear expression from a summation.
        
        Args:
            terms: Iterable of variables or linear expressions to sum.
            
        Returns:
            Linear expression representing the sum of all terms.
        """
        return mip.xsum(terms)

    def _validate_params(self) -> None:
        check_scalar(
            x=self.max_gap,
            name="max_gap",
            lb=0,
            var_type=float,
            include_boundaries=False,
        )
        check_scalar(
            x=self.max_gap_abs,
            name="max_gap_abs",
            lb=0,
            var_type=float,
            include_boundaries=False,
        )
        check_scalar(
            x=self.infeasibility_tol,
            name="infeasibility_tol",
            var_type=float,
            lb=0,
            include_boundaries=False,
        )
        if self.smoothing is not None:
            check_scalar(
                x=self.smoothing,
                name="smoothing",
                var_type=float,
                lb=0,
                ub=1,
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
    def _validate_bounds(lb: float | int, ub: float | int, var_type: str) -> tuple[float | int, float | int]:
        """Validate and normalize variable bounds.
        
        Args:
            lb: Lower bound value.
            ub: Upper bound value.
            var_type: Variable type ('C', 'I', or 'B').
            
        Returns:
            Tuple of (validated_lb, validated_ub).
        """
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
                ub=mip.INF,
                lb=lb,
                include_boundaries=False,
            )
        return lb, ub
