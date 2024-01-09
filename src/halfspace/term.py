from typing import Union, Callable, Optional

import mip
import numpy as np

QueryPoint = dict[mip.Var, float]
Input = Union[float, list[float], np.ndarray]
Var = Union[mip.Var, list[mip.Var], mip.LinExprTensor]
Fun = Callable[[Input], float]
Grad = Callable[[Input], Union[float, np.ndarray]]


class NonlinearTerm:

    def __init__(
        self,
        var: Var,
        func: Fun,
        grad: Grad,
        step_size: float = 1e-6,
        feasibility_tol: float = 1e-4,
        is_constraint: bool = False,
        name: str = "",
    ):
        """Nonlinear term constructor.

        Args:
            var:
            func:
            grad:
            step_size: float
            feasibility_tol: float,
            is_constraint: bool, default=False
            name: str, default=''
        """
        self.var = var
        self.func = func
        self.grad = grad
        self.step_size = step_size
        self.tol = feasibility_tol
        self.is_constraint = is_constraint
        self.name = name
        self._validate()

    def __call__(self, query_point: QueryPoint, return_grad: bool = False) -> Union[float, tuple[float, np.ndarray]]:
        x = self._get_input(query_point=query_point)
        value = self._evaluate_func(x=x)
        if return_grad:
            return value, self._evaluate_grad(x=x)
        return value

    @property
    def is_multivariable(self) -> bool:
        return not isinstance(self.var, mip.Var)

    def generate_cut(self, query_point: QueryPoint = None) -> Optional[mip.LinExpr]:

        # Evaluate term
        fun, grad = self(query_point=query_point, return_grad=True)

        # If term is non-violated constraint, no cut is generated
        if self.is_constraint and fun <= self.tol:
            return

        # Otherwise, make expression for cut
        x = self._get_input(query_point=query_point)
        if self.is_multivariable:
            return mip.xsum(grad * (np.array(self.var) - x)) + fun
        return grad * (self.var - x) + fun

    def _validate(self) -> None:
        pass

    def _get_input(self, query_point: QueryPoint) -> Input:
        if self.is_multivariable:
            return np.array([query_point[var] for var in self.var])
        return query_point[self.var]

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



