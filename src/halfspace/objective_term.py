from typing import Union, Callable

import mip
import numpy as np

Input = Union[float, list[float], np.ndarray]
Fun = Callable[[Input], float]
Grad = Callable[[Input], Union[float, np.ndarray]]
Variables = Union[mip.Var, list[mip.Var], mip.LinExprTensor]


class ObjectiveTerm:

    def __init__(
        self,
        var: Variables,
        func: Fun,
        grad: Grad,
        step_size: float,
        name: str = ""
    ):
        """Objective term constructor.

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
        self.name = name
        self._validate()

    @property
    def x(self) -> Union[float, np.ndarray]:
        """Get the variable value(s) of the incumbent solution."""
        if isinstance(self.var, mip.Var):
            return float(self.var.x)
        return np.array([float(var.x) for var in self.var])

    @property
    def value(self) -> float:
        """Get the objective term value of the incumbent solution."""
        return self._evaluate_func(x=self.x)

    @property
    def is_multivariable(self) -> bool:
        return not isinstance(self.var, mip.Var)

    def generate_cut(self, x: Input = None) -> mip.LinExpr:
        """Generate a cutting plane for the objective term."""
        if x is None:
            x = self.x
        fun = self._evaluate_func(x=x)
        grad = self._evaluate_grad(x=x)
        if isinstance(self.var, mip.Var):
            return grad * (self.var - x) + fun
        return mip.xsum(grad * (np.array(self.var) - x)) + fun

    def _validate(self) -> None:
        pass

    def _evaluate_func(self, x: Input) -> float:
        """Evaluate the objective term value."""
        if isinstance(self.var, (mip.Var, mip.LinExprTensor)):
            return self.func(x)
        if isinstance(self.var, list):
            return self.func(*x)
        raise NotImplementedError

    def _evaluate_grad(self, x: Input) -> np.ndarray:
        """Evaluate the objective term gradient."""
        if self.grad is None:
            return self._approximate_grad(x=x)
        if isinstance(self.var, (mip.Var, mip.LinExprTensor)):
            return self.grad(x)
        if isinstance(self.var, list):
            return self.grad(*x)
        raise NotImplementedError

    def _approximate_grad(self, x: Input) -> Union[float, np.ndarray]:
        """Approximate the gradient of the function at point using the central finite difference method."""
        if isinstance(self.var, mip.Var):
            return (
                self._evaluate_func(x=x + self.step_size / 2)
                - self._evaluate_func(x=x - self.step_size / 2)
            ) / self.step_size
        else:
            indexes = np.arange(len(x))
            return np.array([
                (
                    self._evaluate_func(x=x + self.step_size / 2 * (indexes == i))
                    - self._evaluate_func(x=x - self.step_size / 2 * (indexes == i))
                ) / self.step_size
                for i in indexes
            ])


