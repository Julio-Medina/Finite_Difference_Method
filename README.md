# Finite Difference Method for Elliptic PDEs

This repository contains a numerical implementation of the **finite difference method** for elliptic partial differential equations, with emphasis on the two-dimensional Poisson equation on a rectangular domain.

I originally wrote this project during my **BSc. in Physics**. The goal was not only to solve a PDE numerically, but also to make explicit the full path from the mathematical formulation to the discrete system, and from there to the algorithm used to solve it.

The repository includes the Python implementation and a LaTeX report with the mathematical derivation, algorithms, and examples.

> **GitHub Markdown note:** formulas are written using fenced `math` blocks instead of inline dollar signs. This is more reliable in GitHub READMEs, especially for longer equations, matrices, and indexed variables.

---

## Problem solved in this project

The main equation considered is the two-dimensional Poisson equation:

```math
\nabla^2 u(x,y)
\equiv
\frac{\partial^2 u}{\partial x^2}(x,y)
+
\frac{\partial^2 u}{\partial y^2}(x,y)
=
f(x,y).
```

The equation is solved on a rectangular domain:

```math
R = \{(x,y) \mid a < x < b,\; c < y < d\}.
```

The boundary condition is Dirichlet type:

```math
u(x,y) = g(x,y), \qquad (x,y) \in \partial R.
```

The goal is to approximate the unknown values of `u` at the interior points of a grid.

---

## Discretization of the rectangular domain

The rectangle is divided into `n` subintervals in the `x` direction and `m` subintervals in the `y` direction. The mesh sizes are:

```math
h = \frac{b-a}{n},
\qquad
k = \frac{d-c}{m}.
```

The grid points are:

```math
x_i = a + ih,
\qquad
 i = 0,1,2,\ldots,n,
```

```math
y_j = c + jk,
\qquad
j = 0,1,2,\ldots,m.
```

The interior points are:

```math
(x_i,y_j),
\qquad
 i = 1,2,\ldots,n-1,
\qquad
 j = 1,2,\ldots,m-1.
```

At those points, the numerical approximation is denoted by:

```math
w_{ij} \approx u(x_i,y_j).
```

---

## Centered finite-difference formulas

The second derivative with respect to `x` is approximated by the centered-difference formula:

```math
\frac{\partial^2 u}{\partial x^2}(x_i,y_j)
=
\frac{u(x_{i+1},y_j)-2u(x_i,y_j)+u(x_{i-1},y_j)}{h^2}
-
\frac{h^2}{12}
\frac{\partial^4 u}{\partial x^4}(\xi_i,y_j),
```

where:

```math
\xi_i \in (x_{i-1},x_{i+1}).
```

The second derivative with respect to `y` is approximated by:

```math
\frac{\partial^2 u}{\partial y^2}(x_i,y_j)
=
\frac{u(x_i,y_{j+1})-2u(x_i,y_j)+u(x_i,y_{j-1})}{k^2}
-
\frac{k^2}{12}
\frac{\partial^4 u}{\partial y^4}(x_i,\eta_j),
```

where:

```math
\eta_j \in (y_{j-1},y_{j+1}).
```

Substituting these approximations into the Poisson equation gives:

```math
\begin{aligned}
&\frac{u(x_{i+1},y_j)-2u(x_i,y_j)+u(x_{i-1},y_j)}{h^2}
+
\frac{u(x_i,y_{j+1})-2u(x_i,y_j)+u(x_i,y_{j-1})}{k^2}
\\[4pt]
&=
f(x_i,y_j)
+
\frac{h^2}{12}\frac{\partial^4 u}{\partial x^4}(\xi_i,y_j)
+
\frac{k^2}{12}\frac{\partial^4 u}{\partial y^4}(x_i,\eta_j).
\end{aligned}
```

Therefore, the local truncation error is of order:

```math
O(h^2+k^2).
```

---

## Finite-difference equation

Ignoring the truncation terms and writing the unknown values as `w_ij`, the finite-difference equation used in the implementation is:

```math
2\left[\left(\frac{h}{k}\right)^2 + 1\right]w_{ij}
-
\left(w_{i+1,j}+w_{i-1,j}\right)
-
\left(\frac{h}{k}\right)^2
\left(w_{i,j+1}+w_{i,j-1}\right)
=
-h^2 f(x_i,y_j).
```

This equation is applied for:

```math
i = 1,2,\ldots,n-1,
\qquad
j = 1,2,\ldots,m-1.
```

The boundary values are inserted using:

```math
w_{0j}=g(x_0,y_j),
\qquad
w_{nj}=g(x_n,y_j),
\qquad
j=0,1,\ldots,m,
```

and:

```math
w_{i0}=g(x_i,y_0),
\qquad
w_{im}=g(x_i,y_m),
\qquad
i=1,2,\ldots,n-1.
```

After applying the boundary conditions, the problem becomes a linear system:

```math
A w = b.
```

---

## Relabeling the grid points

The two-dimensional unknowns `w_ij` are relabeled into a one-dimensional vector. In the mathematical report, the one-indexed relabeling convention is:

```math
P_l=(x_i,y_j),
\qquad
w_l=w_{ij},
```

with:

```math
l(i,j)=i+(m-1-j)(n-1),
```

for:

```math
i=1,2,\ldots,n-1,
\qquad
j=1,2,\ldots,m-1.
```

The Python implementation uses zero-indexed loops. The equivalent zero-indexed function is:

```math
L(i,j)=(i+1)+(m-1-(j+1))(n-1)-1.
```

This is implemented in the code as:

```python
def l(i, j, n, m):
    return (i + 1) + (m - 1 - (j + 1)) * (n - 1) - 1
```

This ordering labels the interior grid points from left to right and from top to bottom. The resulting coefficient matrix is banded and block tridiagonal.

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
| `code/crout_factorization_generalization.py` | Implements a Crout-type algorithm for block tridiagonal matrices. |
| `Finite_Difference_Method.tex` | Mathematical report with the derivation, algorithms, and examples. |
| `Finite_Difference_Method.pdf` | Compiled version of the report. |
| `code/error_table.csv` | Example output comparing numerical and analytical values. |

---

## Linear solver

The finite-difference discretization produces a block tridiagonal matrix. Instead of treating the matrix as completely unstructured, the project includes a direct solver based on a generalization of Crout factorization for block tridiagonal systems.

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

The report also discusses the SOR method as an iterative alternative. For a linear system:

```math
A x = b,
```

SOR updates the `i`-th component by:

```math
x_i^{(k+1)}
=
(1-\omega)x_i^{(k)}
+
\frac{\omega}{a_{ii}}
\left(
b_i
-
\sum_{j=1}^{i-1}a_{ij}x_j^{(k+1)}
-
\sum_{j=i+1}^{n}a_{ij}x_j^{(k)}
\right).
```

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

The script builds the linear system, solves it using both `numpy.linalg.solve` and the Crout generalization, and writes an error table to CSV.

---

## Main functions

### Finite-difference system builder

```python
finite_difference_linear_system(a, b, c, d, n, m, f, g)
```

Builds the linear system associated with the finite-difference discretization.

| Argument | Meaning |
|---|---|
| `a, b, c, d` | Rectangular domain limits. |
| `n, m` | Number of grid subdivisions in the `x` and `y` directions. |
| `f` | Right-hand side of the Poisson equation. |
| `g` | Boundary-condition function. |

Returns:

```python
A, w, x, y
```

where `A` is the coefficient matrix, `w` is the constant vector, and `x`, `y` are the grid coordinates.

### Crout generalization solver

```python
Crout_generalization(A, K, n)
```

Solves a block tridiagonal linear system using the Crout-type block factorization implemented in the repository.

### Error table generator

```python
error_table(n, m, x, y, w, u)
```

Compares the numerical approximation with an analytical solution and exports the result as `error_table.csv`.

---

## Example 1: Laplace equation on a square

The first example in the report solves:

```math
\frac{\partial^2 u}{\partial x^2}(x,y)
+
\frac{\partial^2 u}{\partial y^2}(x,y)
=
0,
```

on:

```math
R=\{(x,y)\mid 0<x<0.5,\;0<y<0.5\}.
```

The boundary conditions are:

```math
u(0,y)=0,
\qquad
u(x,0)=0,
\qquad
u(x,0.5)=200x,
\qquad
u(0.5,y)=200y.
```

For `n=m=4`, one has `h=k`, so the finite-difference equation becomes:

```math
4w_{ij}
-
w_{i+1,j}
-
w_{i-1,j}
-
w_{i,j-1}
-
w_{i,j+1}
=
0.
```

For example, at `i=1`, `j=1`:

```math
4w_{1,1}-w_{2,1}-w_{0,1}-w_{1,0}-w_{1,2}=0.
```

Because the boundary values give:

```math
w_{0,1}=0,
\qquad
w_{1,0}=0,
```

this reduces to:

```math
4w_{1,1}-w_{2,1}-w_{1,2}=0.
```

Using the one-indexed relabeling function from the report:

```math
l(1,1)=7,
\qquad
l(2,1)=8,
\qquad
l(1,2)=4,
```

so the same equation becomes:

```math
4w_7-w_8-w_4=0.
```

The coefficient matrix is:

```math
A =
\begin{bmatrix}
4 & -1 & 0 & -1 & 0 & 0 & 0 & 0 & 0 \\
-1 & 4 & -1 & 0 & -1 & 0 & 0 & 0 & 0 \\
0 & -1 & 4 & 0 & 0 & -1 & 0 & 0 & 0 \\
-1 & 0 & 0 & 4 & -1 & 0 & -1 & 0 & 0 \\
0 & -1 & 0 & -1 & 4 & -1 & 0 & -1 & 0 \\
0 & 0 & -1 & 0 & -1 & 4 & 0 & 0 & -1 \\
0 & 0 & 0 & -1 & 0 & 0 & 4 & -1 & 0 \\
0 & 0 & 0 & 0 & -1 & 0 & -1 & 4 & -1 \\
0 & 0 & 0 & 0 & 0 & -1 & 0 & -1 & 4
\end{bmatrix}.
```

The constant vector is:

```math
b =
\begin{bmatrix}
25 \\
50 \\
150 \\
0 \\
0 \\
50 \\
0 \\
0 \\
25
\end{bmatrix}.
```

Solving:

```math
Aw=b
```

gives:

```math
w=
\begin{bmatrix}
18.75 \\
37.5 \\
56.25 \\
12.5 \\
25 \\
37.5 \\
6.25 \\
12.5 \\
18.75
\end{bmatrix}.
```

---

## Example 2: Poisson equation with logarithmic boundary data

The report presents a second example with:

```math
\frac{\partial^2 u}{\partial x^2}(x,y)
+
\frac{\partial^2 u}{\partial y^2}(x,y)
=
4.
```

The numerical system shown in the report has the coefficient matrix:

```math
A=
\begin{bmatrix}
4 & -1 & -1 & 0 \\
-1 & 4 & 0 & -1 \\
-1 & 0 & 4 & -1 \\
0 & -1 & -1 & 4
\end{bmatrix}.
```

The constant vector shown is:

```math
b=
\begin{bmatrix}
1.165 \\
2.774 \\
-0.444 \\
1.165
\end{bmatrix}.
```

Solving:

```math
Aw=b
```

gives approximately:

```math
w=
\begin{bmatrix}
0.5825 \\
0.9849 \\
0.1801 \\
0.5825
\end{bmatrix}.
```

However, this example has notation/domain inconsistencies in the original report. I list them explicitly in the section **Formula corrections and issues found** below instead of hiding them.

---

## Example 3: Comparison against an analytical solution

The third example solves:

```math
\frac{\partial^2 u}{\partial x^2}(x,y)
+
\frac{\partial^2 u}{\partial y^2}(x,y)
=
x e^y,
```

on:

```math
R=\{(x,y)\mid 0<x<2,\;0<y<1\}.
```

The boundary conditions are:

```math
u(0,y)=0,
\qquad
u(x,0)=x,
\qquad
u(x,1)=e x,
\qquad
u(2,y)=2e^y.
```

The analytical solution is:

```math
u(x,y)=xe^y.
```

This makes it possible to compute the absolute error at each interior grid point:

```math
\left|u(x_i,y_j)-w_{ij}\right|.
```

The generated output table is an error table. The first two columns identify the interior grid point, and the remaining columns compare the numerical approximation with the analytical solution. I use HTML subscripts in this table because they render reliably on GitHub, unlike raw LaTeX inside Markdown tables.

<table>
  <thead>
    <tr>
      <th>Column</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><i>i</i>, <i>j</i></td>
      <td>Interior grid indices.</td>
    </tr>
    <tr>
      <td><i>x</i><sub>i</sub>, <i>y</i><sub>j</sub></td>
      <td>Coordinates of the interior grid point.</td>
    </tr>
    <tr>
      <td><i>w</i><sub>ij</sub></td>
      <td>Numerical approximation at the grid point.</td>
    </tr>
    <tr>
      <td><i>u</i>(<i>x</i><sub>i</sub>, <i>y</i><sub>j</sub>)</td>
      <td>Analytical value at the same grid point.</td>
    </tr>
    <tr>
      <td>|<i>u</i>(<i>x</i><sub>i</sub>, <i>y</i><sub>j</sub>) − <i>w</i><sub>ij</sub>|</td>
      <td>Absolute error between the analytical and numerical values.</td>
    </tr>
  </tbody>
</table>

Mathematically, the comparison made in the table is:

```math
w_{ij} \approx u(x_i,y_j).
```

and the reported error is:

```math
\left|u(x_i,y_j)-w_{ij}\right|.
```

---

## Formula corrections and issues found

These are the formulas or statements from the original report that I found problematic while preparing the README. I am listing them here instead of removing the surrounding material.

### 1. Grid definition typo

The report writes the `y` grid as:

```math
y=a+jk.
```

That should be:

```math
y_j=c+jk.
```

The reason is that the vertical interval is `[c,d]`, not `[a,b]`.

### 2. Mixed index in the centered-difference formulas

Several formulas in the report write terms such as:

```math
u(x_i,y_i).
```

For a two-dimensional grid point indexed by `i` and `j`, this should be:

```math
u(x_i,y_j).
```

The same correction applies to the second derivatives:

```math
\frac{\partial^2u}{\partial x^2}(x_i,y_j),
\qquad
\frac{\partial^2u}{\partial y^2}(x_i,y_j).
```

### 3. Index of the Taylor remainder in the `y` direction

The report states the interval using `eta_i` and `y_i` notation. The correct `y`-direction remainder should use the `j` index:

```math
\eta_j \in (y_{j-1},y_{j+1}).
```

### 4. Missing neighbor in the stencil list

The report lists the points involved in the stencil as:

```math
(x_{i-1},y_j),
\quad
(x_i,y_j),
\quad
(x_i,y_{j-1}),
\quad
(x_i,y_{j+1}).
```

But the five-point stencil also includes:

```math
(x_{i+1},y_j).
```

So the complete stencil is:

```math
(x_i,y_j),
\quad
(x_{i-1},y_j),
\quad
(x_{i+1},y_j),
\quad
(x_i,y_{j-1}),
\quad
(x_i,y_{j+1}).
```

### 5. Example 1 variable list skips `w_6`

The report says there are nine equations for:

```math
w_1,w_2,w_3,w_4,w_5,w_7,w_8,w_9.
```

That list skips `w_6`. The complete list should be:

```math
w_1,w_2,w_3,w_4,w_5,w_6,w_7,w_8,w_9.
```

### 6. Example 2 domain and boundary notation are inconsistent

The report states the domain as:

```math
R=\{(x,y)\mid 0<x<1,\;0<y<2\}.
```

But the boundary conditions shown use boundaries at `x=1`, `x=2`, `y=0`, and `y=1`, which correspond more naturally to a rectangle like:

```math
1<x<2,
\qquad
0<y<1.
```

This also matches the values used in the current Python script:

```python
a = 1.0
b = 2.0
c = 0.0
d = 1.0
n = 3
m = 3
```

### 7. Example 2 boundary conditions appear swapped/mislabeled

The report gives boundary conditions involving expressions such as:

```math
u(1,y)=\ln(y^2+1),
\qquad
u(x,0)=2\ln x,
\qquad
u(x,1)=\ln(x^2+1),
\qquad
u(2,y)=\ln(y^2+4).
```

But the Python function currently assigns boundary values in a different pattern. This section should be reviewed before treating Example 2 as a polished textbook-style example.

### 8. Duplicate algorithm label/caption in the report

The SOR algorithm section reuses the Crout caption/label in the original report. The SOR algorithm should have its own label and caption, for example:

```math
\text{Finite differences using the SOR relaxation method.}
```

---

## Why this project matters to me

This project is a good example of the kind of numerical work I enjoy: starting from the mathematical formulation, deriving the discrete approximation, writing the algorithm, and then validating the result computationally.

It connects several ideas that are important in computational physics and numerical analysis:

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
BSc. in Physics / Data Scientist  
GitHub: [Julio-Medina](https://github.com/Julio-Medina)
