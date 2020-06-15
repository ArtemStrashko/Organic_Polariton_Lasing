import numpy as np

# funtion for tensor multiplication of matrices A and B and then transforming a tensor to a 2d matrix
def tenz_dot_to_matr(A, B):
    '''Returns tensor product of matrices A and B flattened to a 2d matrix.
    Example: tenz_dot_to_matr(np.identity(2), np.array([[1,1],[2,2]])) => 
    array([[ 1.,  1.,  0.,  0.],
           [ 2.,  2.,  0.,  0.],
           [ 0.,  0.,  1.,  1.],
           [ 0.,  0.,  2.,  2.]])'''
    return np.hstack(np.hstack( np.tensordot(A, B, axes = 0) ))



# Hermitian conjugate
def dag(A):
    '''Returns Hermitian conjugate of a matrix
    Example: dag(np.array([[1j, 1],[2, -1J]])) = 
    array([[ 0.-1.j,  2.-0.j],
           [ 1.-0.j,  0.+1.j]])'''
    return np.transpose(np.conjugate(A))



# Scalar matrix product
def scal_matr_prod(A,B):
    '''Returns trace of a product of two matrices'''
    return np.trace( np.dot(A, B) )



# commutator
def commut(H, rho):
    com = np.dot(H, rho) - np.dot(rho, H)
    return com



# triple matrix product
def trip_pr(A,B,C):
    prod = np.dot( A, np.dot(B, C) )
    return prod



# Lindblad operator
def Lindblad(X, rho):
    lind = 0.5 * ( 2 * trip_pr( X, rho, dag(X) ) - trip_pr( dag(X), X, rho ) - trip_pr( rho, dag(X), X ) )
    return lind






def exp_coef(N, eps, om_v, S, g_light_mat, g_non_rwa, lam):
    
    N = 2 * N # total number of states
    
    eps = 0.5 * eps

    n_lam = lam.shape[0]
    # Pauli matrices
    sig_x = np.array([[0,1],[1,0]], dtype = complex)
    sig_y = np.array([[0, -1j],[1j, 0]])
    sig_z = np.array([[1, 0],[0, -1]], dtype = complex)
    sig_plus = 0.5 * (sig_x + 1j * sig_y)
    sig_minus = 0.5 * (sig_x - 1j * sig_y)
    proj_up = 0.5 * (np.identity(2) + sig_z)
    sig_pm = 2.0 * np.dot(sig_plus, sig_minus)


    # phonon annihilation operator b 
    b = np.zeros(shape=(int(N/2), int(N/2)))
    for i in range(int(N/2) - 1):
        b[i, i + 1] = np.sqrt(i+1)


    # phonon displacement operator
    x = dag(b) + b


    # phonon number operator matrix nb = b^{\dagger} b
    nb = np.dot(dag(b), b) 


    # A anb B Hamiltonian matrices
    A_ham = (
            eps * tenz_dot_to_matr( sig_z, np.identity(int(N/2)) ) + 
        
            om_v * ( 
                    tenz_dot_to_matr( np.identity(2), nb ) + 
                    
                    np.sqrt(S) * tenz_dot_to_matr(sig_z, x ) 
                    #  np.sqrt(S) * tenz_dot_to_matr(sig_pm, x)
                    ) 
            )


    B_ham = ( 
        
              g_light_mat * tenz_dot_to_matr( sig_minus, np.identity(int(N/2)) ) + 
             
              g_non_rwa   * tenz_dot_to_matr( sig_plus , np.identity(int(N/2)) )
            )    
    
    
    
    # Expansion coefficients for matrices A_ham and B_ham - A_i and B_i
    sig_z_jump = tenz_dot_to_matr( sig_z, np.identity(int(N/2)) )
    A = np.empty( [n_lam], dtype = np.complex128 )
    B = np.empty( [n_lam], dtype = np.complex128 )
    phon_num_exp_coef = np.empty( [n_lam], dtype = np.complex128 )
    sig_z_exp_coef = np.empty( [n_lam], dtype = np.complex128 )
    
    
    for i in range(n_lam):
        A[i] = 0.5 * scal_matr_prod(A_ham, lam[i])
        B[i] = 0.5 * scal_matr_prod(B_ham, lam[i])
        phon_num_exp_coef[i] = 0.5 * scal_matr_prod( tenz_dot_to_matr(np.identity(2), nb), lam[i])
        sig_z_exp_coef[i] = 0.5 * scal_matr_prod( sig_z_jump, lam[i])
        
    return A, B, sig_z_exp_coef






def rates(Gam_up, Gam_down, Gam_z, gam_phon, lam, N, T, om_v, S):
    
    N = 2 * N # total number of states

    n_lam = lam.shape[0]
    sig_x = np.array([[0,1],[1,0]], dtype = complex)
    sig_y = np.array([[0, -1j],[1j, 0]])
    sig_z = np.array([[1, 0],[0, -1]], dtype = complex)
    sig_plus = 0.5 * (sig_x + 1j * sig_y)
    sig_minus = 0.5 * (sig_x - 1j * sig_y)
    b = np.zeros(shape=(int(N/2), int(N/2)))
    for i in range(int(N/2)):
        for j in range(int(N/2)):
            if j == i + 1:
                b[i,j] = np.sqrt(j)

    # "Old" jump operators O_i
    sig_plus_jump = tenz_dot_to_matr( sig_plus, np.identity(int(N/2)) )
    sig_minus_jump = dag(sig_plus_jump)
    sig_z_jump = tenz_dot_to_matr( sig_z, np.identity(int(N/2)) )
    #b_dag_jump = tenz_dot_to_matr( np.identity(2), dag(b) )
    b_dag_jump = tenz_dot_to_matr( np.identity(2), dag(b) ) - np.sqrt(S) * sig_z_jump
    #b_jump = tenz_dot_to_matr( np.identity(2), b ) 
    b_jump = tenz_dot_to_matr( np.identity(2), b ) - np.sqrt(S) * sig_z_jump
    

    O = np.array([sig_plus_jump, sig_minus_jump, b_dag_jump, b_jump, sig_z_jump])


    
    n_mean_phon = 1.0 / (np.exp(om_v / T) - 1) # mean number delocalized phonons
    phon_up = gam_phon * n_mean_phon # phonon gamma up
    phon_down = gam_phon * (n_mean_phon + 1) # phonon gamma down
    
    # "Old" rates \Gamma_i
    old_rates = np.array([Gam_up, Gam_down, phon_up, phon_down, Gam_z])


    # new "rates" \gamma_i^{\mu}
    mu_num = O.shape[0] # number of old jump operators
    c = np.empty( [mu_num, n_lam], dtype = complex )

    for mu in range(mu_num):
        for i in range(n_lam):
            c[mu, i] = 0.5 * scal_matr_prod(O[mu], lam[i]) 



    gam_mu_i = np.empty( [mu_num, n_lam], dtype = complex )

    for mu in range(mu_num):
        for i in range(n_lam):
            gam_mu_i[mu, i] = np.sqrt(old_rates[mu]) * c[mu, i] 
            
    return gam_mu_i





def equat_coeff(gam_mu_i, f, g, A, N):
    zeta = 1j * f + g
    
    N = 2 * N # total number of states

    # calculating xi (below I use m for mu)
    gam = gam_mu_i
    gam_star = np.conjugate(gam_mu_i)

    xi = (1j * ( np.einsum("jk, ijkp ->ip",
                          np.einsum("mj, mk -> jk", gam, gam_star), 
                          np.einsum("ijl, klp -> ijkp", f, zeta)) + 

                np.einsum("jk, kijp ->ip",
                          np.einsum("mj, mk -> jk", gam, gam_star), 
                          np.einsum("kil, ljp -> kijp", f, zeta)) 
               ) 
                  +           

                2 * np.einsum("ijp, j -> ip", f, A)   

            )



    # calculating psi (using a for alpha)
    psi = ( np.einsum("ikq, qla -> ikla", f, zeta) + 
            np.einsum("liq, kqa -> ikla", f, zeta)
          )


    # calculating phi 
    phi = - ( ( (4j)/(N) ) * 

             np.einsum("lk, ikl -> i", 
                      np.einsum("ml, mk -> lk", gam, gam_star), f)

            )

    # calculating eta
    eta = - 1j * (   

             np.einsum("lk, ikla -> ia", 
                      np.einsum("ml, mk -> lk", gam, gam_star), psi )

            )
    
    return zeta, xi, psi, phi, eta


