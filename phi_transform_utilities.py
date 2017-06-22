"""
A some simple utilities for dealing with complex matrix variables in the framework of cvxpy

The phi transform is defined in "Uncertainty--Shape Identification in the Frequency Domain" 
by Gray C. Thomas and Luis Sentis. It is our name for the transform that takes a complex matrix
A + Bj and represents it as a real block matrix [[A, -B], [B, A]]
"""
import numpy as np
import cvxpy as cvx

def phi_image_structure(matrix, cons):
    # guarantees that a real matrix is in the image of phi, and thus in the domain of phi inverse.
    n = matrix.size[0]/2
    m = matrix.size[1]/2
    cons.append(matrix[0:n, m:2*m] == -matrix[n:2*n, 0:m])
    cons.append(matrix[0:n, 0:m] == matrix[n:2*n, m:2*m])
    return matrix

def cvx_block_matrix(array):
    return cvx.vstack([cvx.hstack(row) for row in array])

def inverse_phi(phi_A, enf_tol=False, enf_hrm=False):
    # options are enforce(d) tolerance (usage: enf_tol=1e-4) and enforce hermietian structure (enf_hrm)
    n = phi_A.shape[0]/2
    m = phi_A.shape[1]/2
    A_r1 = phi_A[0:n,0:m]
    A_r2 = phi_A[n:2*n,m:2*m]
    A_i1 = -phi_A[0:n,m:2*m]
    A_i2 = phi_A[n:2*n,0:m]
    if enf_tol:
        # assert that the matrix is close to being in the domain of phi inverse
        assert(np.linalg.norm(A_r1-A_r2,'fro')<enf_tol)
        assert(np.linalg.norm(A_i1-A_i2,'fro')<enf_tol)
        if enf_hrm:
            # assert that the matrix is close to the phi-image of a hermetian matrix
            assert(np.linalg.norm(A_r1-A_r1.T, 'fro')<enf_tol)
            assert(np.linalg.norm(A_r2-A_r2.T, 'fro')<enf_tol)
            assert(np.linalg.norm(A_i1+A_i2.T, 'fro')<enf_tol)
            assert(np.linalg.norm(A_i1+A_i2.T, 'fro')<enf_tol)
    # cast to image and apply inverse phi
    A = 0.5*A_r1 + 0.5*A_r2 + complex(0,1)*(0.5*A_i1 + 0.5*A_i2)
    if enf_hrm:
        # cast to hermetian
        A = 0.5*A + 0.5*A.H
    return A

def vec_phi(np_vec):
    # recall that the phi transform behaves differently on vectors
    return np.hstack([np_vec.real, np_vec.imag]).reshape((-1, 1))