"""This module can be used to perform basic checks on the functions
that you have to code.

The fact that the tests here are passed does not imply that your
functions are correctly coded. You may expand the code below to
perform more detailed checks on your functions. When grading this
assignment, a complete set of checks will be performed.

PLEASE DO NOT SUBMIT THIS FILE. 

Syntax:

To check problem <P> write

$ python3 checks.py <P>

in the command line, where <P> is "1a", "1b", "1c", etc.

Please report any mistake in this file to daniel.romero@uia.no

"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Replace `lastname_givenname` with the name of your file.
from anni_isaackobby import Functions as F

tol = 1e-5

############################################################
# Section A. Functions to run checks.
############################################################


def check_problem_1a():

    N = 5
    x = np.random.normal(size=(N, ))
    y = np.random.normal(size=(N, ))
    z = F.equilateral(x, y)

    # Format checks
    assert_vector(z)
    assert len(z) == len(x), "Invalid format"

    # Value checks
    dxy = np.linalg.norm(x - y)
    dxz = np.linalg.norm(x - z)
    dyz = np.linalg.norm(y - z)

    if (not approx_equal(dxy, dxz)) or (not approx_equal(
            dxy, dyz)) or (not approx_equal(dxz, dyz)):
        print("Test failed")
    else:
        print("Success: Test passed")


def check_problem_1b():

    c = np.random.normal(size=(2, ))
    R = np.random.random()
    x = np.random.normal(size=(2, ), scale=2.)

    y = F.nearest_on_circle(c, R, x)

    # Format checks
    assert_vector(y)
    assert len(y) == 2, "Invalid format"

    # Value checks
    if np.linalg.norm(y - c) > R + 2 * tol:
        print("Fail: The returned point is not in the circle.")
        return

    # Write additional checks (optional)


def check_problem_1c():

    h = np.random.random()
    w = h + np.random.random()
    alpha = 90 * np.random.random()
    n = np.random.randint(low=1, high=10)

    x_n = F.reflections_in_tunnel(w, h, alpha, n)

    # Format checks
    assert_vector(x_n)
    assert len(x_n) == 2, "Invalid format"

    # Figure
    reflections = np.array(
        [F.reflections_in_tunnel(w, h, alpha, ind) for ind in range(1, n + 1)])
    max_x = np.max(reflections[:, 0])
    max_y = np.max([w, np.max(reflections[:, 1])])
    plt.plot([0, max_x], [0, 0], "b")
    plt.plot([0, max_x], [max_y, max_y], "b")
    plt.plot([0], [h], "kx")
    path = np.concatenate(([[0, h]], reflections), axis=0)
    plt.plot(path[:, 0], path[:, 1], "r")
    print(reflections)
    plt.show()

    # Write additional checks (optional)


def check_problem_1d():

    c_x = np.random.random()
    R = c_x + np.random.random()
    alpha = 45 * np.random.random()

    L = F.reflection_on_circle(c_x, R, alpha)

    # Format checks
    assert_number(L)

    # Write additional checks (optional)


def check_problem_2a():

    L = np.random.randint(1, 6)
    x_vals = []
    for _ in range(L):
        x_vals += [np.random.normal()] * np.random.randint(1, 6)
    x = np.array(x_vals)

    S, u = F.repeated_entries(x)

    # Format checks
    assert_vector(u)
    assert S.shape == (len(x), len(u))

    # Value checks
    assert np.linalg.norm(x - S @ u) < L * tol, "Fail: x != S*u"

    vals_S = set(np.reshape(S, [-1]))
    for val in vals_S:
        if (val != 0) and (val != 1):
            print("S contains entries that are different from 0 and 1")
            return

    assert len(u) == L

    if len(u) > 1:
        assert all(
            (u[1:] - u[:-1]) > 0), "Fail: u must have increasing entries."

    print("Success!")


def check_problem_2b():

    N = np.random.randint(1, 5)
    M = N + np.random.randint(1, 5)
    B = np.random.normal(size=(M, N))
    v = np.random.normal(size=(M, ))

    v_par, v_perp = F.orthogonal_components(B, v)

    # Format checks
    assert_vector(v_par)
    assert_vector(v_perp)
    assert len(v_par) == M, "Invalid format"
    assert len(v_perp) == M, "Invalid format"

    # Value checks
    if np.linalg.norm(v_par + v_perp - v) > M * tol:
        print("Fail: v is not equal to v_par + v_perp")
        return

    B_v_par = np.concatenate((B, v_par[:, None]), axis=1)
    if np.linalg.matrix_rank(B) != np.linalg.matrix_rank(B_v_par):
        print("Fail: v_par is not a linear combination of the columns of B")
        return

    if np.linalg.norm(B.T @ v_perp[:, None]) > N * tol:
        print("Fail: v_perp is not perpendicular to the columns of B")
        return

    print("Success")


def check_problem_2c():

    N = np.random.randint(1, 5)
    M = N + np.random.randint(1, 5)
    B = np.random.normal(size=(M, N))

    v = F.orthogonal_vector(B)

    # Format checks
    assert_vector(v)
    assert len(v) == M, "Invalid format"

    # Value checks
    if np.linalg.norm(v) < M * tol:
        print("Fail: v cannot be the zero vector")
        return

    if np.linalg.norm(B.T @ v[:, None]) > N * tol:
        print("Fail: v is not perpendicular to the columns of B")
        return

    print("Success")


def check_problem_3a():

    N = np.random.randint(2, 5)
    M = np.random.randint(2, 5)
    P = np.random.random(size=(M, N))
    P /= np.sum(P)

    p_x = F.marginal_x(P)

    # Format checks
    assert_vector(p_x)
    assert len(p_x) == M, "Invalid format"

    # Write additional checks (optional)


def check_problem_3b():

    N = np.random.randint(2, 5)
    M = np.random.randint(2, 5)
    P = np.random.random(size=(M, N))
    P /= np.sum(P)

    e = F.expectation_y(P)

    # Format checks
    assert_number(e)

    # Write additional checks (optional)


def check_problem_3c():

    N = np.random.randint(2, 5)
    M = np.random.randint(2, 5)
    P = np.random.random(size=(M, N))
    P /= np.sum(P)
    n = np.random.randint(1, N + 1)

    p_x_y = F.conditional_x(P, n)

    # Format checks
    assert_vector(p_x_y)
    assert len(p_x_y) == M, "Invalid format"

    # Write additional checks (optional)


def check_problem_3d():

    N = np.random.randint(1, 5)
    M = np.random.randint(1, 5)
    P = np.random.random(size=(M, N))
    P /= np.sum(P)
    n = np.random.randint(1, N + 1)

    e = F.conditional_expectation_x(P, n)

    # Format checks
    assert_number(e)

    # Write additional checks (optional)


############################################################
# Section B. Auxiliary code.
############################################################


def assert_vector(v):

    if not isinstance(v, np.ndarray):
        print("The output is not an np.array.")
        quit()

    if v.ndim != 1:
        print("The output has not ndim==1")
        quit()


def assert_number(num):

    if isinstance(num, np.ndarray):
        if num.size == 1:
            return

    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    if not all(hasattr(num, attr) for attr in attrs):
        print("The output is not of a numeric type")
        quit()


def approx_equal(a, b):

    if isinstance(a, np.ndarray):
        N = a.size
    else:
        N = 1
    return (np.linalg.norm(a - b) <= N * tol)


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print('Usage: python3 checks.py  <problem>')
        print('       with <problem> = "1a","1b","1c",etc')
    else:
        str_problem = sys.argv[1]
        str_fun = "check_problem_" + str_problem

        if str_fun not in dir():
            raise ValueError(
                f"There is no function to check problem {str_problem}.")
        # Execute the corresponding check function in Sec. A.
        locals()[str_fun]()
