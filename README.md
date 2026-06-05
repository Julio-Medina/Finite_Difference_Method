# Finite Difference Method for Elliptic PDEs

This repository contains a numerical implementation of the **finite difference method** for elliptic partial differential equations, with emphasis on the two-dimensional Poisson equation on a rectangular domain.

I originally wrote this project as part of my graduate work in physics. The goal was not only to solve a PDE numerically, but also to make explicit the full path from the mathematical discretization to the matrix system and then to the algorithm used to solve it.

The main problem considered is

$$
\nabla^2 u(x,y) = \frac{\partial^2 u}{\partial x^2}(x,y) + \frac{\partial^2 u}{\partial y^2}(x,y) = f(x,y),
$$

on a rectangular region

$$
R = \{(x,y) \mid a < x < b,\; c < y < d\},
$$

with Dirichlet boundary conditions

$$
u(x,y) = g(x,y) \quad \text{on the boundary of } R.
$$

The repository includes both the Python implementation and a LaTeX report explaining the mathematical derivation.

---

## What this project does

The code builds the finite-difference approximation of the Poisson equation by discretizing a rectangular domain into an \(n \times m\) grid. The interior points of the grid become the unknowns of a linear system.

For each interior point \((x_i, y_j)\), the method uses the standard centered-difference approximation

$$
2\left[\left(\frac{h}{k}\right)^2 + 1\right]w_{ij}
- \left(w_{i+1,j} + w_{i-1,j}\right)
- \left(\frac{h}{k}\right)^2\left(w_{i,j+1} + w_{i,j-1}\right)
= -h^2 f(x_i,y_j),
$$

where \(w_{ij}\) approximates the exact solution \(u(x_i,y_j)\).

After applying the boundary conditions, the discretized problem becomes a linear system

$$
Aw = b.
$$

A key point of the implementation is that the unknowns are relabeled so that the coefficient matrix has a **block tridiagonal structure**. This makes it natural to solve the system using a generalization of the Crout factorization algorithm for block tridiagonal matrices.

---

## Repository structure

```text
.
├── Finite_Difference_Method.tex          # Main LaTeX report
├── Finite_Difference_Method.pdf          # Compiled report
├── lattice.png                           # Grid diagram used in the report
├── Diagram1.dia                          # Source diagram file
├── README.md                             # Project documentation
└── code/
    ├── finite_difference_linear_system.py
    ├── crout_factorization_generalization.py
    ├── Linear_system.csv
    ├── constant_vector.csv
    └── error_table.csv
```

The most important files are:

| File | Description |
|---|---|
| `code/finite_difference_linear_system.py` | Builds the finite-difference linear system for the Poisson equation and runs an example. |
| `code/crout_factorization_generalization.py` | Implements the Crout generalization for block tridiagonal matrices. |
| `Finite_Difference_Method.tex` | Mathematical report with the derivation, algorithms, and examples. |
| `Finite_Difference_Method.pdf` | Compiled version of the report. |
| `code/error_table.csv` | Example output comparing numerical and analytical values. |

---

## Numerical method

The finite difference method is applied to the Poisson equation by replacing the second derivatives with centered finite differences. The step sizes are

$$
h = \frac{b-a}{n}, \qquad k = \frac{d-c}{m}.
$$

The truncation error of this discretization is of order

$$
O(h^2 + k^2).
$$

The implementation then maps every interior grid point \((x_i,y_j)\) to a single linear index. This transforms the two-dimensional grid problem into a matrix problem while preserving the sparse block structure produced by the finite-difference stencil.

The relabeling function used in the code is

$$
l(i,j) = (i+1) + (m-1-(j+1))(n-1) - 1,
$$

which is the zero-indexed version used directly in Python.

---

## Linear solver

The resulting matrix is block tridiagonal. Instead of treating the system as a completely general dense linear system, the repository includes an implementation of a **generalized Crout factorization** for block tridiagonal matrices.

The solver is implemented in:

```text
code/crout_factorization_generalization.py
```

The core function is:

```python
Crout_generalization(A, K, n)
```

where:

| Argument | Meaning |
|---|---|
| `A` | Coefficient matrix of the linear system. |
| `K` | Constant vector. |
| `n` | Block size of the block tridiagonal matrix. |

The report also discusses the SOR method as an alternative iterative approach, although the current Python implementation focuses on the finite-difference system construction and the Crout-based direct solver.

---

## How to run the code

Clone the repository and move into the code directory:

```bash
git clone https://github.com/Julio-Medina/Finite_Difference_Method.git
cd Finite_Difference_Method/code
```

Install the required Python packages:

```bash
pip install numpy pandas
```

Run the finite-difference example:

```bash
python finite_difference_linear_system.py
```

The script defines a rectangular domain, builds the linear system, solves it with both `numpy.linalg.solve` and the Crout generalization, and writes an error table to CSV.

---

## Main functions

### `finite_difference_linear_system`

```python
finite_difference_linear_system(a, b, c, d, n, m, f, g)
```

Builds the matrix system associated with the finite-difference discretization.

| Argument | Meaning |
|---|---|
| `a, b, c, d` | Rectangular domain limits. |
| `n, m` | Number of grid subdivisions in the x and y directions. |
| `f` | Right-hand side of the Poisson equation. |
| `g` | Boundary-condition function. |

Returns:

```python
A, w, x, y
```

where `A` is the coefficient matrix, `w` is the constant vector, and `x`, `y` are the grid coordinates.

### `Crout_generalization`

```python
Crout_generalization(A, K, n)
```

Solves a block tridiagonal linear system using the generalized Crout factorization.

### `error_table`

```python
error_table(n, m, x, y, w, u)
```

Compares the numerical approximation with an analytical solution and exports the result as `error_table.csv`.

---

## Example problem

One of the examples in the report solves

$$
\frac{\partial^2 u}{\partial x^2}(x,y)
+
\frac{\partial^2 u}{\partial y^2}(x,y)
=
x e^y,
$$

with boundary conditions chosen so that the analytical solution is

$$
u(x,y) = x e^y.
$$

This makes it possible to compare the numerical approximation against the exact solution and compute the absolute error at each interior grid point.

The output table has the form:

| Column | Meaning |
|---|---|
| `i`, `j` | Interior grid indices. |
| `x_i`, `y_j` | Coordinates of the grid point. |
| `w_ij` | Numerical approximation. |
| `u(x_i,y_j)` | Analytical value. |
| `\|u(x_i,y_j)-w_ij\|` | Absolute error. |

---

## Why this project matters to me

This project is a good example of the kind of numerical work I enjoy: starting from the mathematical formulation, deriving the discrete approximation, writing the algorithm, and then validating the result computationally.

It also connects several ideas that are important in computational physics and numerical analysis:

- finite-difference discretization of PDEs,
- construction of structured linear systems,
- boundary-value problems,
- direct solvers for block tridiagonal matrices,
- comparison between numerical and analytical solutions.

The current version is intentionally simple and educational. A future refactor could turn it into a cleaner Python package with tests, examples, plotting utilities, and support for additional solvers.

---

## Possible improvements

Some natural next steps for the repository are:

- separate the example execution from the library functions,
- add unit tests for the finite-difference matrix construction,
- add tests comparing the Crout solver against `numpy.linalg.solve`,
- avoid explicit matrix inverses inside the Crout implementation,
- add plotting utilities for the numerical solution surface,
- add a command-line interface for choosing the domain, grid size, and boundary conditions,
- implement the SOR method discussed in the report,
- organize the code as an installable Python package.

---

## References

The derivation and algorithms are based mainly on:

- Richard L. Burden and J. Douglas Faires, *Numerical Analysis*, Ninth Edition.
- Richard S. Varga, *Matrix Iterative Analysis*, Second Edition.

---

## Author

**Julio A. Medina**  
BSc. in Physics/Data Scientist  
GitHub: [Julio-Medina](https://github.com/Julio-Medina)
