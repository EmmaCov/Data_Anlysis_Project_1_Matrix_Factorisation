# Matrix Factorization using constraint programming
# Goal: given a matrix R, find matrices I and U so that IxU' = Y ~= R
# We define the I and U coefficients as variables
# We define constraints as inequalities for each Y coefficient: it must not be too far from its R equivalent
# Warning: we assume all R coefficients are in [0, 1]

import numpy as np
from ortools.constraint_solver import pywrapcp


def matrix_factorization_cp(R: np.array, k: int = 1, tolerance: float = 0.1, precision: int = 2):
    m, n = R.shape
    factor = 10 ** precision
    amplified_R = R * factor

    solver = pywrapcp.Solver("Factorization")

    # define variables
    I_vars = [[solver.IntVar(-factor, factor, f"I_{i}_{j}") for j in range(k)] for i in range(m)]
    U_vars = [[solver.IntVar(-factor, factor, f"U_{i}_{j}") for j in range(k)] for i in range(n)]

    # define constraints
    for i in range(m):
        for j in range(n):
            R_i_j = amplified_R[i, j]
            predicted_R_i_j = sum(I_vars[i][l] * U_vars[j][l] for l in range(k))
            solver.Add(predicted_R_i_j >= round(R_i_j * (1 - tolerance)))
            solver.Add(predicted_R_i_j <= round(R_i_j * (1 + tolerance)))

    all_vars = [v for line in I_vars for v in line] + [v for line in U_vars for v in line]
    db = solver.Phase(all_vars, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
    solver.NewSearch(db)

    if solver.NextSolution():
        I = np.array([[I_vars[i][j].Value() for j in range(k)] for i in range(m)])
        U = np.array([[U_vars[i][j].Value() for j in range(k)] for i in range(n)])
        solver.EndSearch()
        I = I / factor
        return I, U

    return None


def frobenius_error(R, I, U):
    return np.linalg.norm(R - I.dot(np.transpose(U))) / np.linalg.norm(R) * 100

def create_fake_R(m, n, k, digits):
    I = np.trunc(np.random.rand(m, k) * 10**digits) / 10**digits
    U = np.trunc(np.random.rand(n, k) * 10**digits) / 10**digits
    R = I.dot(U.transpose())
    noise = np.random.normal(0, 10**-digits, R.shape)
    return R + noise

if __name__ == '__main__':
    m, n = 6, 6
    k = 1
    tolerance = 0.4
    digits_precision = 3
    np.random.seed(42)

    R = create_fake_R(m, n, k, digits_precision)
    factor = matrix_factorization_cp(R, k, tolerance, digits_precision)

    if factor is not None:
        I, U = factor
        print("Solution found with error:", round(frobenius_error(R, I, U)), "%")
    else:
        print("No solution found.")
