import numpy as np 

def numSUM(mat):
    """
    Function for counting the number of zeros element in a matrix `mat`
    Tolerance is 1e-4.
    """
    zeros = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            if abs(mat[i,j]) < 1e-4:
                zeros += 1
    return zeros

from qiskit.visualization import array_to_latex

# the original matrix
matA = np.array([
            [1.0, 1.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 2.0, 1.0, 1.0],
            [1.0, 2.0, 0.0, 4.0, 1.0],
            [2.0, 2.0, 1.0, 4.0, 6.0]])

sizeA = matA.shape[0] # size matrix, 5

max_iteration = 100 
iteration_counter = 0
delta_sum_zero = 0
sum_zero = 0
Flag = False

while Flag == False and iteration_counter < max_iteration:
    Q, R = np.linalg.qr(matA) # using built-in function. I hope we're allowed to do this
    matA = R@Q # Flip QR => RQ to for similarity transformation
    iteration_counter += 1
    # below are just stopping criterion, described in the report
    if iteration_counter % sizeA == 0:
        delta_sum_zero = numSUM(matA) - sum_zero
        if delta_sum_zero == 0:
            Flag = True
    sum_zero = numSUM(matA)
    
array_to_latex(matA)

# finding immediate non-zero elements below diagonals
non_zeros = []
zeros = []

for i in range(sizeA-1):
    if abs(matA[i+1][i]) > 1e-3:
        non_zeros.append(i)
submats = []

# construction of 2x2 sub-matrices
for index in non_zeros:
    submats.append(np.array([
        [matA[index][index], matA[index][index+1]],
        [matA[index+1][index], matA[index+1][index+1]]
    ]))
    
import cmath

#solving an ordinary polynomials of order 2

def characteristics_polynomial(mat):
    a = 1.0
    b = -(mat[0][0]+mat[1][1])
    c = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
    x = (-b + cmath.sqrt(b**2-4*a*c))/2
    return x

eigvals = []
eigvals.append(matA[0][0])

for mat in submats:
    eigval = characteristics_polynomial(mat)
    eigvals.append(eigval)
    eigvals.append(np.conjugate(eigval))

eigvals

# An attempt to find eigenvectors using Gaussian elimination, but unsuccessful

matA = np.array([
            [1.0, 1.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 2.0, 1.0, 1.0],
            [1.0, 2.0, 0.0, 4.0, 1.0],
            [2.0, 2.0, 1.0, 4.0, 6.0]])

A_tilde = matA

for i in range(sizeA):
    A_tilde[i][i] = matA[i][i] - eigvals[0]

zeros_vec = np.zeros([sizeA, 1])
eigen_vec1 = Gauss_Seidel(A_tilde, zeros_vec, tol=1e-8)
eigen_vec2 = np.linalg.solve(A_tilde, zeros_vec)

eigen_vec1 