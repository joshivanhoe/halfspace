# `halfspace`

`halfspace` is a light-weight Python module for modelling and solving convex optimization problems of the form:

$$
\begin{align}
\min ~& f_0(x), \\
\text{s.t.} ~& f_i(x) \leq 0, && \forall i=1,...,m, \\
& a_i^\top x = b_i,  && \forall i=1,...,p, \\
& l \leq x \leq u,
\end{align}
$$

where $f_0,...,f_m$ are convex functions. It is built on top of the high-performance Python `mip` module and uses a cutting plane algorithm to solve problems to provable optimality. Conviently, this approach extends to mix-integer problems, in which integrality constraints also apply to a subset of the decision variables.

## Quick start

You can install `halfspace` using `pip` as follows:

```bash
pip install halfspace
```

The modelling syntax for `halfspace` closely follows that of the `mip` module. As an illustrative example, let's consider the toy problem:

$$
\begin{align}
\min_{x,y,z} ~& (x - 1)^2 + \exp(0.2x + y) + \sum_{i=1}^5 i z_i, \\
\text{s.t.}  ~& \sum_{i=1}^5 z_i \leq y, \\
& x^2 + y^2 \leq 1.25^2, \\
& x\in[0, 1], \\
& y\in \\{0, 1 \\}, \\
& z\in[0, 1]^5. \\
\end{align}
$$

This can be implemented as follows:

```python
from halfspace import Model

# Initialize model
model = Model()

# Define variables
x = model.add_var(lb=0, ub=1)  # add a variable
y = model.add_var(var_type="B")  # add a binary variable
z = model.add_var_tensor(shape=(5,), lb=0, ub=1)  # add a tensor of variables

# Define objective terms
model.add_objective_term(var=x, func=lambda x: (x - 1) ** 2)  # add an objective term for one variable
model.add_objective_term(var=[x, y], func=lambda x, y: np.exp(0.2 * x + y))  # add an objective term for multiple variables
model.add_objective_term(var=z, func=lambda z: -sum((i + 1) * z[i] for i in range(5)))  # add an objective term for a tensor of variables

# Define constraints
model.add_linear_constr(model.sum(z) <= y) # add a linear constraint
model.add_nonlinear_constr(var=(x, y), func=lambda x, y: x**2 + y**2 - 1.25 ** 2)  # add a nonlinear constraint

# Solve model
status = model.optimize()
print(status, model.best_objective)
```

## How it works

To be added

## Development

To be added
