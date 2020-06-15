import numpy as np
from Coefficients_rwa_non_rwa import *
from MF_equations__with_A_squared import *
import time
from pylab import figure, plot, show
import matplotlib.pyplot as plt
from numpy.linalg import inv

# number of phonon states starting from | 0 > 
N = 4 

# Hamiltonian constants
om_c = 1.0    # photon freq
eps = 1.0     # excit freq
S = 0.1 # exciton-phonon coupling
om_v = 0.2    # phon freq

# number of molecules
N_m = 1.0

# Lindblad equation parameters
Gam_down = 1e-4  # excit decay
kappa = 1e-4     # photon leakage
Gam_z = 0.03     # exciton dephasing 
T = 0.025        # phon bath temp
gam_phon = 0.02 # phonon-bath coupling


# coupling and pumping arrays
g_light_mat_array = np.linspace(0.0, 1.0, 500)
Gam_up_array = np.linspace(0.0, 1.6e-4, 500)


# defining GGM lambda matrices and structure factors f and g
from GGM import GGM_matr

lx = GGM_matr(N)[2]
lz = GGM_matr(N)[3]
lam = GGM_matr(N)[4]
f = GGM_matr(N)[5]
g = GGM_matr(N)[6]


ns_stable = np.zeros( ( len(Gam_up_array), len(g_light_mat_array) ), dtype=int)

# calculating coefficients A_i and B_i  (requires Hamiltonian parameters)
ec = exp_coef(N, eps, om_v, S, 1.0, 1.0, lam)
A = ec[0] # doesn't depend on colupling, so calculated for coupling 1.0 (can put any constants)
B = ec[1] # it's just proportional to coupling, so calculate it here for coupling 1

i = j = 0

start_time = time.time() 
for Gam_up in Gam_up_array:
    
    j = 0
        
    # Calculating rates gamma_i^{\mu}  (requires Lindblad parameters)
    gam_mu_i = rates(Gam_up, Gam_down, Gam_z, gam_phon, lam, N, T, om_v, S)

    # calculating zeta, xi and psi  (requires Hamiltonian and Lindblad parameters)
    zeta, xi, psi, phi, eta = equat_coeff(gam_mu_i, f, g, A, N)

    # normal state specification
    A_vec = - (2j / N) * np.einsum('mj, mk, ijk -> i', gam_mu_i, np.conjugate(gam_mu_i), f)
    xi_inv_matr = inv(xi)
    lam_ns = np.real( np.dot(xi_inv_matr, A_vec) )
   
    for g_light_mat in g_light_mat_array:  
        
        BB = B * g_light_mat # coef B is proportional to coupling

        # build stability matrix with A^2 term
        M = M_stability_with_A_squared(om_c, kappa, BB, f, xi, lam_ns, N_m, g_light_mat, eps)
        
        # stability matrix without A^2
        #M = M_stability(om_c, kappa, BB, f, xi, lam_ns, N_m)

        # eigenvalues
        max_eig, n_unstab = stable_eigs(M)

        ns_stable[i, j] = n_unstab

        j = j + 1
    i = i + 1
    print( i, "out of ", str(len(Gam_up_array)) )
        
print("--- %s seconds ---" % (time.time() - start_time))        






# plotting/saving results
ft = 14

Y, X = np.meshgrid(g_light_mat_array, Gam_up_array / Gam_down)

plt.figure(figsize=(3.5, 2.5))
plt.contourf(Y, X, ns_stable, colors = ('white', 'black'), levels = [0,1,4])
plt.xlabel('$g \sqrt{N_m}$', size=ft)
plt.ylabel('$ \Gamma_{\\uparrow} / \Gamma_{\\downarrow} $', size=ft)
plt.xticks(fontsize=ft, rotation=0)
plt.yticks(fontsize=ft, rotation=0)
plt.text(0.05, 1.2, '$\omega_c = $' + str(om_c), fontsize = 18, color='w')

x = [0.0, 1.0]
y = [0.0, 1.6]
plt.xticks(np.arange(min(x), max(x)+0.1, 0.2))
plt.yticks(np.arange(min(y), max(y)+0.1, 0.4))

#plt.savefig('phd_10_with_A2.svg', bbox_inches='tight', transparent=True)

show(block=False)

