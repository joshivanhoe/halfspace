from typing import Union, Callable, Optional, Iterable

import mip
import numpy as np

QueryPoint = dict[mip.Var, float]
Input = Union[float, Iterable[float], np.ndarray]
Var = Union[mip.Var, Iterable[mip.Var], mip.LinExprTensor]
Func = Callable[[Input], float]
Grad = Callable[[Input], Union[float, np.ndarray]]


class ConvexTerm:

    def __init__(
        self,
        var: Var,
        func: Func,
        grad: Optional[Grad] = None,
        step_size: float = 1e-6,
        name: str = "",
    ):
        """Convex term constructor.

        Args:
            var: mip.Var or iterable of mip.Var or mip.LinExprTensor
                The variable(s) included in the term. This can be provided in the form of a single  variable, an
                iterable of multiple variables or a variable tensor.
            func: callable mapping input(s) to float
                The function representing the term. For a single variable term, there should be a single float argument
                multivariable term, there
            grad: callable input to array, default=`None`
                A function for computing the term's gradient. If `None`, then the gradient is approximated numerically
                using the central finite difference method.
            step_size: float, default=`1e-6`
                The step size used for numerical gradient approximation. If `grad` is provided, then this argument is
                ignored.
            name: str, default=''
                The name for the term.
        """
        self.var = var
        self.func = func
        self.grad = grad
        self.step_size = step_size
        self.name = name

    def __call__(self, query_point: QueryPoint, return_grad: bool = False) -> Union[float, tuple[float, np.ndarray]]:
        """Evaluate the term and (optionally) its gradient.

        Args:
            query_point: dict mapping mip.Var to float
                The query point at which the term is evaluated.
            return_grad: bool, default=`False`
                Whether to return the term's gradient.

        Returns: float or tuple of float and array
            If `return_grad=False`, then only the value of the term is returned. Conversely, if `return_grad=True`,
            then a tuple is returned where the first element is the term's value and the second element is the term's
            gradient.
        """
        x = self._get_input(query_point=query_point)
        value = self._evaluate_func(x=x)
        if return_grad:
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

        Returns: mip.LinExpr
            The linear constraint representing the cutting plane.
        """
        fun, grad = self(query_point=query_point, return_grad=True)
        x = self._get_input(query_point=query_point)
        if self.is_multivariable:
            return mip.xsum(grad * (np.array(self.var) - x)) + fun
        return grad * (self.var - x) + fun

    def _get_input(self, query_point: QueryPoint) -> Input:
        if self.is_multivariable:
            return np.array([query_point[var] for var in self.var])
        return query_point[self.var]

    def _evaluate_func(self, x: Input) -> float:
        """Evaluate the function value."""
        if isinstance(self.var, (mip.Var, mip.LinExprTensor)):
            return self.func(x)
        if isinstance(self.var, Iterable):
            return self.func(*x)
        raise TypeError(f"Input of type '{type(x)}' not supported.")

    def _evaluate_grad(self, x: Input) -> np.ndarray:
        """Evaluate the gradient."""
        if self.grad is None:
            return self._approximate_grad(x=x)
        if isinstance(self.var, (mip.Var, mip.LinExprTensor)):
            return self.grad(x)
        if isinstance(self.var, Iterable):
            return self.grad(*x)
        raise TypeError(f"Input of type '{type(x)}' not supported.")

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
