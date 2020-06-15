import numpy as np
from Coefficients_rwa_non_rwa import *
from MF_equations__with_A_squared import *
import numpy.linalg as linalg
from numpy.linalg import inv, norm
import time


def unstab_bound_vertical_calc(N, eps, om_v, S, g_light_mat, g_non_rwa, lam, phot_freq_stab, 
                              Gam_up_array_stab, Gam_down, Gam_z, gam_phon, T, f, g, kappa, N_m):
    
    # calculating coefficients A_i and B_i  (requires Hamiltonian parameters)
    ec = exp_coef(N, eps, om_v, S, g_light_mat, g_non_rwa, lam)
    A = ec[0]
    B = ec[1]

    # empty arrays 
    eigenvectors_unstab = np.zeros([len(phot_freq_stab), len(lam[::, 0, 0]) + 2], dtype = complex )
    gam_up_position = np.zeros([len(phot_freq_stab)])
    xi_pumping_array = np.zeros([np.shape(lam)[0],np.shape(lam)[0],len(Gam_up_array_stab)], dtype = complex)
    lam_ns_pumping_array = np.zeros([np.shape(lam)[0],len(Gam_up_array_stab)], dtype = complex)
    
    # precalculations
    jj = 0
    for Gam_up in Gam_up_array_stab:
        
        # Calculating rates gamma_i^{\mu}  (requires Lindblad parameters)
        gam_mu_i = rates(Gam_up, Gam_down, Gam_z, gam_phon, lam, N, T, om_v, S)
        gam = gam_mu_i
        gam_star = np.conjugate(gam_mu_i)

        # calculating zeta, xi and psi  (requires Hamiltonian and Lindblad parameters)
        zeta, xi_pumping_array[:,:,jj], psi, phi, eta = equat_coeff(gam_mu_i, f, g, A, N)

        # normal state calculation
        A_vec = - (2j / N) * np.einsum('mj, mk, ijk -> i', gam, gam_star, f)
        xi_inv_matr = inv(xi_pumping_array[:,:,jj])
        lam_ns_pumping_array[:,jj] = np.real( np.dot(xi_inv_matr, A_vec) )
                                 
        jj = jj + 1
    

    # stability calculations
    i = 0
    for om_c in phot_freq_stab:
        
        print('freq step is ' + str(i+1) + ' out of ' + str(len(phot_freq_stab)) + ', freq is ' + str(np.round(om_c, 3))   )
        
        start_time = time.time()  

        j = 0
        for Gam_up in Gam_up_array_stab:

            xi = xi_pumping_array[:,:,j]
            lam_ns = lam_ns_pumping_array[:,j]

            # stability matrix 
            M = M_stability_with_A_squared(om_c, kappa, B, f, xi, lam_ns, N_m, g_light_mat, eps)

            # get eigenvalues and assosiated eigenvectors
            eigenValues, eigenVectors = linalg.eig(M)

            # take only eigenvalues with positive imaginary parts
            filter_ev = np.imag(eigenValues)>0
            eigenValues = eigenValues[filter_ev]
            eigenVectors = eigenVectors[:,filter_ev]

            # sort the according to eigenvalues' real parts from smallest to largest
            idx = eigenValues.argsort()#[::-1]   
            eigenValues = eigenValues[idx]
            eigenVectors = eigenVectors[:,idx]

            # take only eigenvalues with positive real part (unstable ones)
            if np.real(eigenValues[-1]) > 0:
                eigenvectors_unstab[i,:] = eigenVectors[:,-1]
                gam_up_position[i] = Gam_up
                break
            else:
                pass

            j = j + 1

        i = i + 1
        
        print("--- %s seconds ---" % (time.time() - start_time))   
        
        
        
    # extract photonic part
    phot_part_of_eigenv = eigenvectors_unstab[:, 0:2]

    # extract matter part to build density matrices
    mat_part_of_eigenv = eigenvectors_unstab[:, 2:]

    # build density matrix
    rho_tot = np.zeros(  [len(phot_freq_stab), int(2*N), int(2*N) ], dtype = complex  )

    for ii in range(len(phot_freq_stab)):
        rho_tot[ii, :, :] = 0.5 * np.einsum('i, ijk -> jk', mat_part_of_eigenv[ii, :], lam) 
            
            
    return rho_tot, gam_up_position
