from typing import Union, Callable, Optional

import mip
import numpy as np

Input = Union[float, list[float], np.ndarray]
Fun = Callable[[Input], float]
Grad = Callable[[Input], Union[float, np.ndarray]]
Variables = Union[mip.Var, list[mip.Var], mip.LinExprTensor]


class NonlinearTerm:

    def __init__(
        self,
        var: Variables,
        func: Fun,
        grad: Grad,
        step_size: float,
        is_constraint: bool = False,
        name: str = "",
    ):
        """Nonlinear term constructor.

        Args:
            var:
            func:
            grad:
            step_size: float
        """
        self.var = var
        self.func = func
        self.grad = grad
        self.step_size = step_size
        self.is_constraint = is_constraint
        self.name = name
        self._validate()

    @property
    def x(self) -> Union[float, np.ndarray]:
        """Get the variable value(s) of the incumbent solution."""
        if self.is_multivariable:
            return np.array([float(var.x) for var in self.var])
        return float(self.var.x)

    @property
    def value(self) -> float:
        """Get the value of the term corresponding to incumbent solution."""
        return self._evaluate_func(x=self.x)

    @property
    def is_multivariable(self) -> bool:
        return not isinstance(self.var, mip.Var)

    def generate_cut(self, query_point: Optional[dict[mip.Var, float]] = None) -> Optional[mip.LinExpr]:
        """Generate a cutting plane for the term.

        If a cut is not required, return `None`.
        """

        # Extract variable value(s) from query point
        if query_point is None:
            x = self.x
        else:
            if self.is_multivariable:
                x = np.array([query_point[var] for var in self.var])
            else:
                x = query_point[self.var]

        # Evaluate function and gradient
        fun = self._evaluate_func(x=x)
        if self.is_constraint and fun <= 1e-4:
            return
        grad = self._evaluate_grad(x=x)

        # Make expression for cut
        if self.is_multivariable:
            return mip.xsum(grad * (np.array(self.var) - x)) + fun
        return grad * (self.var - x) + fun

    def _validate(self) -> None:
        pass

    def _evaluate_func(self, x: Input) -> float:
        """Evaluate the function value."""
        if isinstance(self.var, (mip.Var, mip.LinExprTensor)):
            return self.func(x)
        if isinstance(self.var, list):
            return self.func(*x)
        raise NotImplementedError

    def _evaluate_grad(self, x: Input) -> np.ndarray:
        """Evaluate the gradient."""
        if self.grad is None:
            return self._approximate_grad(x=x)
        if isinstance(self.var, (mip.Var, mip.LinExprTensor)):
            return self.grad(x)
        if isinstance(self.var, list):
            return self.grad(*x)
        raise NotImplementedError

    def _approximate_grad(self, x: Input) -> Union[float, np.ndarray]:
        """Approximate the gradient of the function at point using the central finite difference method."""
        if self.is_multivariable:
            indexes = np.arange(len(x))
            return np.array([
                (
                        self._evaluate_func(x=x + self.step_size / 2 * (indexes == i))
                        - self._evaluate_func(x=x - self.step_size / 2 * (indexes == i))
                ) / self.step_size
                for i in indexes
            ])
        return (
            self._evaluate_func(x=x + self.step_size / 2)
            - self._evaluate_func(x=x - self.step_size / 2)
        ) / self.step_size



