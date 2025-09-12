"""This module implements the ConvexTerm class.

It provides a modular framework for generating cutting planes.
"""

from typing import Callable, Iterable, Literal, overload

import mip
import numpy as np

from .utils import check_scalar

type QueryPoint = dict[mip.Var, float]
type Var = mip.Var | Iterable[mip.Var] | mip.LinExprTensor
type Input = float | Iterable[float] | np.ndarray
type Func = Callable[[Input], float]
type FuncGrad = Callable[[Input], tuple[float, float | np.ndarray]]
type Grad = Callable[[Input], float | np.ndarray]


class ConvexTerm:
    """Convex term model used for generating cutting planes.

    Attributes:
        var: The variable(s) included in the term. This can be provided in the form of a single  variable, an
            iterable of multiple variables or a variable tensor.
        func: A function for computing the term's value. This function should accept one argument for each
            variable in `var`. If `var` is a variable tensor, then the function should accept a single array.
        grad: A function for computing the term's gradient. This function should accept one argument for each
            variable in `var`. If `var` is a variable tensor, then the function should accept a single array. If
            `None`, then the gradient is approximated numerically using the central finite difference method. If
            `grad` is instead a Boolean and is `True`, then `func` is assumed to return a tuple where the first
            element is the function value and the second element is the gradient. This is useful when the gradient
            is expensive to compute.
        step_size: The step size used for numerical gradient approximation. Must be positive. If `grad` is provided, then this argument
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
            var: The variable(s) included in the term. Can be a single variable, an iterable of variables, or a variable tensor.
            func: The function for computing the term's value.
            grad: The function for computing the term's gradient, or None for numerical approximation, or True if func returns (value, grad).
            step_size: The step size for numerical gradient approximation. Must be positive.
            name: The name for the term.
        """
        check_scalar(
            x=step_size,
            name="step_size",
            var_type=float,
            lb=0,
            include_boundaries=False,
        )
        self.var = var
        self.func = func
        self.grad = grad
        self.step_size = step_size
        self.name = name

    @overload
    def __call__(self, query_point: QueryPoint, return_grad: Literal[False] = False) -> float: ...

    @overload
    def __call__(
        self, query_point: QueryPoint, return_grad: Literal[True] = True
    ) -> tuple[float, float | np.ndarray]: ...

    def __call__(self, query_point: QueryPoint, return_grad: bool = False) -> float | tuple[float, float | np.ndarray]:
        """Evaluate the term and (optionally) its gradient.

        Args:
            query_point: The query point at which the term is evaluated.
            return_grad: Whether to return the term's gradient.

        Returns:
            If `return_grad=False`, then only the value of the term is returned. If `return_grad=True`,
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
        """Check whether the term is multivariable.
        
        Returns:
            True if the term involves multiple variables, False otherwise.
        """
        return not isinstance(self.var, mip.Var)

    def generate_cut(self, query_point: QueryPoint) -> mip.LinExpr:
        """Generate a cutting plane for the term.

        The cutting plane is a linear approximation of the convex term at the given query point,
        valid for all feasible points due to convexity.

        Args:
            query_point: The query point for which the cutting plane is generated.

        Returns:
            A linear expression representing the cutting plane constraint.
        """
        value, grad = self(query_point=query_point, return_grad=True)
        x = self._get_input(query_point=query_point)
        if self.is_multivariable:
            return mip.xsum(grad * (np.array(self.var) - x)) + value
        return grad * (self.var - x) + value

    def _get_input(self, query_point: QueryPoint) -> Input:
        """Extract input values from query point based on variable type.
        
        Args:
            query_point: The query point containing variable values.
            
        Returns:
            Input values in the format expected by the function.
        """
        if self.is_multivariable:
            return np.array([query_point[var] for var in self.var])
        return query_point[self.var]

    def _evaluate_func(self, x: Input) -> float | tuple[float, float | np.ndarray]:
        """Evaluate the function value.

        If `grad=True`, then both the value of the function and its gradient are returned.
        """
        if isinstance(self.var, (mip.Var, mip.LinExprTensor)):
            return self.func(x)
        if isinstance(self.var, Iterable):
            return self.func(*x)
        raise TypeError(f"Input of type '{type(x)}' not supported.")

    def _evaluate_grad(self, x: Input) -> float | np.ndarray:
        """Evaluate the gradient.
        
        Args:
            x: The input values at which to evaluate the gradient.
            
        Returns:
            The gradient value(s).
        """
        if not self.grad:
            return self._approximate_grad(x=x)
        if isinstance(self.var, (mip.Var, mip.LinExprTensor)):
            return self.grad(x)
        if isinstance(self.var, Iterable):
            return self.grad(*x)
        raise TypeError(f"Input of type '{type(x)}' not supported.")

    def _approximate_grad(self, x: Input) -> float | np.ndarray:
        """Approximate the gradient using central finite differences.
        
        Args:
            x: The input values at which to approximate the gradient.
            
        Returns:
            The approximated gradient value(s).
        """
        if self.is_multivariable:
            n_dim = len(x)
            grad = np.zeros(n_dim)
            e = np.eye(n_dim)
            for i in range(n_dim):
                grad[i] = (
                    self._evaluate_func(x=x + self.step_size / 2 * e[i])
                    - self._evaluate_func(x=x - self.step_size / 2 * e[i])
                ) / self.step_size
            return grad
        return (
            self._evaluate_func(x=x + self.step_size / 2) - self._evaluate_func(x=x - self.step_size / 2)
        ) / self.step_size
