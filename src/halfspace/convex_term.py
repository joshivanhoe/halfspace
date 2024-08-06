"""This module implements the ConvexTerm class.

It provides a modular framework for generating cutting planes.
"""

from typing import Callable, Iterable
from typing import TypeAlias, Literal, overload

import mip
import numpy as np

from .utils import standard_basis_vector

QueryPoint: TypeAlias = dict[mip.Var, float]
Var: TypeAlias = mip.Var | Iterable[mip.Var] | mip.LinExprTensor
Input: TypeAlias = float | Iterable[float] | np.ndarray
Func: TypeAlias = Callable[[Input], float]
FuncGrad: TypeAlias = Callable[[Input], tuple[float, float | np.ndarray]]
Grad: TypeAlias = Callable[[Input], float | np.ndarray]


class ConvexTerm:
    """Convex term model used for generating cutting planes.

    Attributes:
        var: The variable(s) included in the term. This can be provided in the form of a single  variable, an
            iterable of multiple variables or a variable tensor.
        func: A function for computing the term's value. This function should except one argument for each
            variable in `var`. If `var` is a variable tensor, then the function should accept a single array.
        grad: A function for computing the term's gradient. This function should except one argument for each
            variable in `var`. If `var` is a variable tensor, then the function should accept a single array. If
            `None`, then the gradient is approximated numerically using the central finite difference method. If
            `grad` is instead a Boolean and is `True`, then `func` is assumed to return a tuple where the first
            element is the function value and the second element is the gradient. This is useful when the gradient
            is expensive to compute.
        step_size: The step size used for numerical gradient approximation. If `grad` is provided, then this argument
            is ignored.
        name: The name for the term.
    """

    def __init__(
        self,
        var: Var,
        func: Func | FuncGrad,
        grad: Grad | bool | None = None,
        step_size: float = 1e-6,
        name: str = "",
    ) -> None:
        """Convex term constructor.

        Args:
            var: The value of the `var` attribute.
            func: The value of the `func` attribute.
            grad: The value of the `grad` attribute.
            step_size: The value of the `step_size` attribute. Must be positive.
            name: The value of the `name` attribute.
        """
        self.var = var
        self.func = func
        self.grad = grad
        self.step_size = step_size
        self.name = name

    @overload
    def __call__(
        self, query_point: QueryPoint, return_grad: Literal[False] = False
    ) -> float:
        ...

    @overload
    def __call__(
        self, query_point: QueryPoint, return_grad: Literal[True] = True
    ) -> tuple[float, float | np.ndarray]:
        ...

    def __call__(
        self, query_point: QueryPoint, return_grad: bool = False
    ) -> float | tuple[float, float | np.ndarray]:
        """Evaluate the term and (optionally) its gradient.

        Args:
            query_point: The query point at which the term is evaluated.
            return_grad: Whether to return the term's gradient.

        Returns:
            If `return_grad=False`, then only the value of the term is returned. Conversely, if `return_grad=True`,
            then a tuple is returned where the first element is the term's value and the second element is the term's
            gradient.
        """
        x = self._get_input(query_point=query_point)
        value = self._evaluate_func(x=x)
        if self.grad is True and not return_grad:
            return value[0]
        elif self.grad is not True and return_grad:
            return value, self._evaluate_grad(x=x)
        return value

    @property
    def is_multivariable(self) -> bool:
        """Check whether the term is multivariable."""
        return not isinstance(self.var, mip.Var)

    def generate_cut(self, query_point: QueryPoint) -> mip.LinExpr:
        """Generate a cutting plane for the term.

        Args:
            query_point: dict mapping mip.Var to float
                The query point for which the cutting plane is generated.

        Returns:
            The linear constraint representing the cutting plane.
        """
        func, grad = self(query_point=query_point, return_grad=True)
        x = self._get_input(query_point=query_point)
        if self.is_multivariable:
            return mip.xsum(grad * (np.array(self.var) - x)) + func
        return grad * (self.var - x) + func

    def _get_input(self, query_point: QueryPoint) -> Input:
        if self.is_multivariable:
            return np.array([query_point[var] for var in self.var])
        return query_point[self.var]

    def _evaluate_func(self, x: Input) -> float | tuple[float, float | np.ndarray]:
        """Evaluate the function value.

        If `grad=True`, then both the value of the function and it's gradient are returned.
        """
        if isinstance(self.var, (mip.Var, mip.LinExprTensor)):
            return self.func(x)
        if isinstance(self.var, Iterable):
            return self.func(*x)
        raise TypeError(f"Input of type '{type(x)}' not supported.")

    def _evaluate_grad(self, x: Input) -> float | np.ndarray:
        """Evaluate the gradient."""
        if not self.grad:
            return self._approximate_grad(x=x)
        if isinstance(self.var, (mip.Var, mip.LinExprTensor)):
            return self.grad(x)
        if isinstance(self.var, Iterable):
            return self.grad(*x)
        raise TypeError(f"Input of type '{type(x)}' not supported.")

    def _approximate_grad(self, x: Input) -> float | np.ndarray:
        """Approximate the gradient of the function at point using the central finite difference method."""
        if self.is_multivariable:
            n_dim = len(x)
            grad = np.zeros(n_dim)
            for i in range(n_dim):
                e_i = standard_basis_vector(i=i, n_dim=n_dim)
                grad[i] = (
                    self._evaluate_func(x=x + self.step_size / 2 * e_i)
                    - self._evaluate_func(x=x - self.step_size / 2 * e_i)
                ) / self.step_size
            return grad
        return (
            self._evaluate_func(x=x + self.step_size / 2)
            - self._evaluate_func(x=x - self.step_size / 2)
        ) / self.step_size
