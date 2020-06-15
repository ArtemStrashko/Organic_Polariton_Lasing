import numpy as np

# tenzor product of matrices to matrix
def tenz_dot_to_matr(A, B):
    '''Returns tensor product of matrices A and B flattened to a 2d matrix.
    Example: tenz_dot_to_matr(np.identity(2), np.array([[1,1],[2,2]])) => 
    array([[ 1.,  1.,  0.,  0.],
           [ 2.,  2.,  0.,  0.],
           [ 0.,  0.,  1.,  1.],
           [ 0.,  0.,  2.,  2.]])'''
    return np.hstack(np.hstack( np.tensordot(A, B, axes = 0) ))

# Scalar matrix product
def scal_matr_prod(A,B):
    '''Returns trace of a product of two matrices'''
    return np.trace( np.dot(A, B) )



# stability matrix without A^2 term
def M_stability(om_c, kappa, B, f, xi, lam_ns, N_m):

    # first and second lines
    first_line = np.append( [0], - 1j * B * N_m )
    first_line = np.append( [-(1j*om_c + kappa/2)], first_line )

    second_line = np.append( [(1j*om_c - kappa/2)], 1j * B * N_m )
    second_line = np.append( [0], second_line )

    high = np.vstack([first_line, second_line])   # sticked two first lines

    # m_i columns
    m_i = 2 * np.einsum("ik, k -> i", 
                        np.einsum("j, ijk -> ik", B, f), 
                        lam_ns)

    # lower part of the full matrix
    mm = np.vstack([m_i,m_i])  # sticked two rows
    low = np.transpose(np.vstack([mm, np.transpose(xi)]))  # sticked transposed xi below and transposed all

    # full matrix
    M = np.vstack([high, low])
    
    return M





# stability matrix with A^2 term
def M_stability_with_A_squared(om_c, kappa, B, f, xi, lam_ns, N_m, g_light_mat, eps):
    
    g = g_light_mat
    om_c_new = om_c * ( 1 + 2 * g**2 * N_m / (om_c * eps) )
    extr = 2 * g**2 * N_m / eps

    # first and second lines
    first_line = np.append( [-1j*extr], - 1j * B * N_m )
    first_line = np.append( [-(1j*om_c_new + kappa/2)], first_line )

    second_line = np.append( [(1j*om_c_new - kappa/2)], 1j * B * N_m )
    second_line = np.append( [1j*extr], second_line )

    high = np.vstack([first_line, second_line])   # sticked two first lines

    # m_i columns
    m_i = 2 * np.einsum("ik, k -> i", 
                        np.einsum("j, ijk -> ik", B, f), 
                        lam_ns)

    # lower part of the full matrix
    mm = np.vstack([m_i,m_i])  # sticked two rows
    low = np.transpose(np.vstack([mm, np.transpose(xi)]))  # sticked transposed xi below and transposed all

    # full matrix
    M = np.vstack([high, low])
    
    return M




# stability matrix
def stable_eigs(M):
    
    from numpy.linalg import eigvals
    from numpy import amax
    
    stabmat = M
    evals = eigvals(stabmat)
   
    n_unstab = 0
    for count in range(len(evals)):
        if evals[count].real>0.0:
            n_unstab+=1
    
    return amax(evals.real), n_unstab

