from sympy import *

x, y = symbols('x y')

f = cos(x*y) + sin(x**4 + y**4) - 1
g = x**2 + y**2 + sin(x*y) - 3

f_x = diff(f, x)
f_y = diff(f, y)
g_x = diff(g, x)
g_y = diff(g, y)

f_x, f_y, g_x, g_y

import numpy as np
import random
import copy

cos = np.cos
sin = np.sin
exp = np.exp

# The jacobian
def J(x):
    return np.array([
        [4*x[0]**3*cos(x[0]**4 + x[1]**4) - x[1]*sin(x[0]*x[1]), -x[0]*sin(x[0]*x[1]) + 4*x[1]**3*cos(x[0]**4 + x[1]**4)],
        [2*x[0] + x[1]*cos(x[0]*x[1]), x[0]*cos(x[0]*x[1]) + 2*x[1]]
    ])

# F(x)
def F(x):
    return np.array([cos(x[0]*x[1]) + sin(x[0]**4 + x[1]**4) - 1, x[0]**2 + x[1]**2 + sin(x[0]*x[1]) - 3])

def Gauss_Seidel(A, b, tol):
    """
    Standard Gauss-Seidel method to solve Ax=b
    """
    
    m = A.shape[0] # A is an n x n matrix
    n = A.shape[1] 
    if (m!=n):
        print(r'Matrix $A$ is not square!')
        return

    # initialize x and x_new 
    x = np.zeros(n)
    x_n = np.zeros(n)

    # counter for number of iterations
    # norm_x is the conventional Euclidean norm
    # for stopping criterion: ||x-x_n||^2
    iteration_counter = 0
    norm_x = 1

    while (abs(norm_x) > tol) and (iteration_counter < 100):
        for i in range(n):
            x_n[i] = b[i]/A[i,i]
            sum = 0
            # Standard Gauss-Seidel update
            for j in range(n):
                if (j<i): sum+=A[i,j]*x_n[j]
                if (j>i): sum+=A[i,j]*x[j]
            x_n[i] -= sum/A[i,i]
        norm_x = np.linalg.norm(x-x_n, ord=2)
        # update guess x
        for i in range(n):
            x[i]=x_n[i]
        iteration_counter += 1
        
    return x

def Newton_Raphson(F, J, x, tol):
    """
    Standard Newton-Raphson method to solve a system of
    nonlinear equation. For simplicity we pre-calculated
    the Jacobian matrix J (hence this function is native
    to the above presented system of nonlinear equations)
    """
    
    Fval = F(x)
    norm_f = np.linalg.norm(Fval, ord=2)
    iteration_counter = 0
    
    while (abs(norm_f) > tol) and (iteration_counter < 100):
        print(f'{iteration_counter}-th iteration!')
        # Delta = Gauss_Seidel(J(x), -Fval, tol) # My own Gauss-Seidel
        Delta = np.linalg.solve(J(x), -Fval) # Scipy built-in solver
        x = x + Delta # Update x
        Fval = F(x) # Re-compute F
        norm_f = np.linalg.norm(Fval, ord=2) # Euclidean norm of F is used for stopping criterion.
        iteration_counter += 1               # If X is sufficiently close to the exact solution, then F(X) ~ 0
        abs_error = (np.linalg.norm(Delta, ord=2))/(np.linalg.norm(x, ord=2)) # Absolute error
        print(f'Absolute error is {abs_error}!')

    return x, iteration_counter        

# Test Gauss-Seidel / Compare with standard np.linalg.solver
A = np.array([[4.,2.,-2.],[4.,9.,-3.],[-2.,-3.,7.]])
b = np.array([2.,8.,10.])

Gauss_Seidel(A, b, tol=1e-8), np.linalg.solve(A, b)

init_guesses = np.array([[1.224, 0.815], [0.815, 1.224], [0.265, 1.500], [1.58, 0.27], 
                         [-0.05, 1.750], [-1.750, 0.05], [-1.225, -0.80], 
                         [-0.8, -1.3], [-1.584, -0.265], [-0.27, -1.59], 
                         [0.05, -1.73], [1.752, -0.072]])

sols = []
for guess in init_guesses:
    sols.append(Newton_Raphson(F, J, guess, tol=1e-10))