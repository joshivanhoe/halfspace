import mip
from typing import Union, Callable, Optional
import numpy as np


Input = Union[float, list[float], np.ndarray]
Fun = Callable[[Input], float]
Grad = Callable[[Input], Union[float, np.ndarray]]
Variables = Union[mip.Var, list[mip.Var], mip.LinExprTensor]


class ObjectiveTerm:

    def __init__(
        self,
        var: Variables,
        fun: Callable,
        grad: Optional[Callable],
        eps: float = 1e-5,
    ):
        self.var = var
        self._eval_fun = fun
        self.grad = grad
        self.eps = eps

    @property
    def x(self) -> Union[float, np.ndarray]:
        if isinstance(self.var, mip.Var):
            return self.var.x
        return np.ndarray([var.x for var in self.var])

    @property
    def value(self) -> float:
        return self._eval_fun(x=self.x)

    def generate_cut(self, x: Input = None) -> mip.LinExpr:
        x = x or self.x
        fun = self._eval_fun(x=x)
        grad = self._eval_grad(x=x)
        if isinstance(self.var, mip.Var):
            return grad * (self.var - x) + fun
        return mip.xsum(grad * (np.array(self.var) - x)) + fun

    def _eval_fun(self, x: Input) -> float:
        if isinstance(self.var, (mip.Var, mip.LinExpr)):
            return self._eval_fun(x)
        if isinstance(self.var, list):
            return self._eval_fun(*x)
        raise NotImplementedError

    def _eval_grad(self, x: Input) -> np.ndarray:
        if self.grad is None:
            return self._approximate_grad(x=x)
        if isinstance(self.var, (mip.Var, mip.LinExpr)):
            return self.grad(x)
        if isinstance(self.var, list):
            return self.grad(*x)
        raise NotImplementedError

    def _approximate_grad(self, x: Input) -> Union[float, np.ndarray]:
        """Approximate the gradient of the function at point using the central finite difference method."""
        if isinstance(self.var, mip.Var):
            return (self._eval_fun(x=x + self.eps / 2) - self._eval_fun(x=x - self.eps / 2)) / self.eps
        else:
            indexes = np.arange(len(x))
            return np.array([
                (
                    self._eval_fun(x=x + self.eps / 2 * (indexes == i))
                    - self._eval_fun(x=x - self.eps / 2 * (indexes == i))
                ) / self.eps
                for i in indexes
            ])

