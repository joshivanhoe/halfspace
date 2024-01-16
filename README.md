# ✨halfspace✨

`halfspace` is an open-source, light-weight Python module for modelling and solving mixed-integer convex optimization problems of the form:

$$
\begin{align}
\min ~& f_0(x), \newline
\text{s.t.} ~& f_i(x) \leq 0, && \forall i=1,...,m, \newline
& a_i^\top x = b_i,  && \forall i=1,...,p, \newline
& x_i \in \mathbb{Z}, && \forall i\in\mathcal{I}, \newline
& l \leq x \leq u,
\end{align}
$$

where $f_0,...,f_m$ are convex functions and $\mathcal{I}$ represents the subset of variables to which integrality constraints apply.
It is built on top of the high-performance Python `mip` module and uses a cutting plane algorithm to solve problems to provable optimality.
This implementation is based on the approach outlined in [Boyd & Vandenberghe (2008)](https://see.stanford.edu/materials/lsocoee364b/05-localization_methods_notes.pdf) - see Chapter 6.


## Quick start

You can install `halfspace` using `pip` as follows:

```bash
pip install halfspace-optimizer
```

The modelling syntax for `halfspace` closely follows that of the `mip` module. As an illustrative example, let's consider the toy problem:

$$
\begin{align}
\min_{x,y,z} ~& (x - 1)^2 + \exp(0.2x + y) + \sum_{i=1}^5 i z_i, \newline
\text{s.t.}  ~& \sum_{i=1}^5 z_i \leq y, \newline
& x^2 + y^2 \leq 1.25^2, \newline
& x\in[0, 1], \newline
& y\in \\{0, 1 \\}, \newline
& z\in[0, 1]^5. \newline
\end{align}
$$

This can be implemented as follows:

```python
import numpy as np

from halfspace import Model

# Initialize model
model = Model()

# Define variables
x = model.add_var(lb=0, ub=1)  # add a variable
y = model.add_var(var_type="B")  # add a binary variable
z = model.add_var_tensor(shape=(5,), lb=0, ub=1)  # add a tensor of variables

# Define objective terms (these are summed to create the objective)
model.add_objective_term(var=x, func=lambda x: (x - 1) ** 2)  # add an objective term for one variable
model.add_objective_term(
    var=[x, y],
    func=lambda x, y: np.exp(0.2 * x + y),
)  # add an objective term for multiple variables
model.add_objective_term(
    var=z,
    func=lambda z: -sum((i + 1) * z[i] for i in range(5)),
)  # add an objective term for a tensor of variables

# Define constraints
model.add_linear_constr(model.sum(z) <= y)  # add a linear constraint
model.add_nonlinear_constr(var=(x, y), func=lambda x, y: x ** 2 + y ** 2 - 1.25 ** 2)  # add a nonlinear constraint

# Set initial query point (optional)
model.start = [(x, 0), (y, 0)] + [(z[i], 0) for i in range(5)]

# Solve model
status = model.optimize()
print(status, model.objective_value)
```

## Troubleshooting

*Q: The solver is converging too slowly. What can I do to improve this?*

A: Here are few tips to improve solve times:
- Improve the initial query point using problem-specific knowledge. In general, the closer the initial query point is to the optimal solution, the fewer iterations should be required for convergence.
- Tune the `smoothing` parameter, which affects query point updates. For some problems, increasing this parameter can dampen query point oscillation, thereby improving convergence. However, if this parameter is set too high, this can also slow down convergence as the updates become too conservative. Consequently, it may be necessary to experiment with a range of values.
- If a feasible solution is 'good enough' for your application, specify the `max_iters_no_improvement` argument when you call the `optimize` method to prevent wasting time closing the optimality gap.
- Review if any integrality constraints can be relaxed.


*Q: The solution to my problem that the solver has output seems wrong. What are some common mistakes that could cause this?*

A: The cutting plane algorithm only works for convex programs and mixed-integer convex programs. Double-check that the formulation of your problem is indeed convex.
Otherwise, if you're computing the gradients analytically, also double-check that the formula is correct. If you believe erroneous behaviour is instead caused by a bug, please submit an [issue](https://github.com/joshivanhoe/halfspace/issues/new).

*Q: How can I view the solver logs in real time?*

Prior to calling `model.optimize()`, change the logging leve as follows:

```python
import logging
logging.getLogger().setLevel(logging.INFO)
```

The logging frequency can be adjusted as desiredusing the model's `log_freq` attribute.

## Development

Clone the repository using `git`:

```bash
git clone https://github.com/joshivanhoe/halfspace
````

Create a fresh virtual environment using `venv`:

```bash
python3.10 -m venv halfspace
```

Alternatively, this can be done using `conda`:

```bash
conda create -n halfspace python=3.10
```

Note that currently Python 3.10 is recommended.
Activate the environment and navigate to the cloned `halfspace` directory. Install a locally editable version of the package using `pip`:

```bash
pip install -e .
```

To check the installation has worked, you can run the tests (with coverage metrics) using `pytest` as follows:

```bash
pytest --cov=halfspace tests/
```

Contributions are welcome! To see our development priorities, refer to the [open issues](https://github.com/joshivanhoe/halfspace/issues).
Please submit a pull request with a clear description of the changes you've made.
