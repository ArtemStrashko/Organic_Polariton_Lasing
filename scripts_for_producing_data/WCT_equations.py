
# coding: utf-8

# In[11]:

import numpy as np
from qutip import *

# emission/absorption spectra
def emis_absorp_freq(N, eps, om_v, S, gam_phon, Gam_z, T):
    
    eps = 0.5 * eps
    
    # System operators
    S_z_ex  = tensor(sigmaz(), identity(N))
    S_m_ex  = tensor(sigmam(), identity(N))
    #S_pm_ex = 2.0 * S_m_ex.dag() * S_m_ex
    b = tensor(identity(2), destroy(N))
    
    # Hamiltonian
    H_N = eps * S_z_ex + om_v * ( b.dag() * b  + np.sqrt(S) * S_z_ex * (b.dag() + b) )
    
    # mean number of bath phonons
    n = 1.0 / (np.exp(om_v / T) - 1.0)
    
    
    tlist = np.linspace(0, 1000, 500000)

    # jump operators
    def jump(Gam_down, Gam_up, Gam_z, gam_phon, n):
        j = (
            [np.sqrt(Gam_down) * S_m_ex, 
             np.sqrt(Gam_up) * S_m_ex.dag(), 
             np.sqrt(Gam_z) * S_z_ex, 
             np.sqrt(gam_phon * (n + 1.0 )) * (b - np.sqrt(S) * S_z_ex) , 
             np.sqrt(gam_phon * n) * (b.dag() - np.sqrt(S) * S_z_ex)]
            )
        return j

    c_ops_N = lambda: jump(Gam_down, Gam_up, Gam_z, gam_phon, n)



    # emission spectrum
    Gam_up = 1e-6 # excit pumping
    Gam_down = 0.0 * Gam_up  # excit decay
    corr_emission = correlation_2op_1t(H_N, None, tlist, c_ops_N(), S_m_ex.dag(), S_m_ex)
    wlist1, spec1 = spectrum_correlation_fft(tlist, corr_emission)



    # absorption spectrum
    Gam_down = 1e-6 # excit decay
    Gam_up = 0.0 * Gam_down  # excit pumping
    corr_absorption = correlation_2op_1t(H_N, None, tlist, c_ops_N(), S_m_ex, S_m_ex.dag())
    corr_absorption=np.conj(corr_absorption)
    wlist2, spec2 = spectrum_correlation_fft(tlist, corr_absorption)
    
    
    return spec1, spec2, wlist2 # g**2 * spec1, g**2 * spec2 - emission and absorption spectra
                                # wlist2=wlist1 - array of frequencies
    
    
    
    
    
    
    
    
    
    
    
    
# WCT dynamic equations (see eq-s 16-19 in PRA 033826)
def diff_WCT_ef(t, state, Gam_up, Gam_down, Gam_p, Gam_m, kappa, N_m):
    
    
    xdot = np.empty(len(state), dtype=complex)
    
    
    
    # number of photons
    a_dag_a = state[0]
    
    # prob to find an excitem molecule
    p = state[1]
    
    # auxiliary functions
    Gam_up_tot = Gam_up + Gam_p * a_dag_a
    Gam_down_tot = Gam_down + Gam_m * (a_dag_a + 1)

    
    
    # building equations
    
    
    # number of photons < a_dag a >
    a_dag_a_dot = ( - kappa * a_dag_a + 
                   
                   N_m * ( 
                            Gam_m * (a_dag_a + 1) * p - 
                            Gam_p * a_dag_a * (1 - p) 
                         ) 

                  )

    
    # prob to find an excited molecule
    p_dot = - Gam_down_tot * p + Gam_up_tot * (1 - p)
    

    
    # and now combine these two derivatives
    
    xdot = np.append(a_dag_a_dot, p_dot)    
    
    
    return xdot











# steady-state solution
def wct_ss(Emis, Abs, kappa, N_m, Gam_up, Gam_down):
    
    c = - (Emis * Gam_up * N_m) / (kappa *  (Emis + Abs))

    b =       ( 
                ( (kappa / N_m) * (Gam_up + Gam_down + Emis) + Abs * Gam_down - Emis * Gam_up ) 

                / 

               (  (kappa / N_m) * ( Abs + Emis ) )
                )



    n_phot = - 0.5 * (  b - np.sqrt( b**2 - 4 * c )  )

    n_phot_small_number = (N_m * Emis * Gam_up) / (kappa * (Gam_up + Gam_down + Emis) )
    
    return n_phot, n_phot_small_number









# a function to find an element index
def index_of_element(array, element):
    j = 0
    epsilon = element / 100.0
    for j in range(len(array)):
        if np.abs( array[j] - element ) > epsilon:
            j = j + 1
        else:
            break
    return j


# In[ ]:



