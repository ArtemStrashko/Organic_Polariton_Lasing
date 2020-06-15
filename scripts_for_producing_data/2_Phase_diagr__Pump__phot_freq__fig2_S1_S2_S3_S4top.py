import numpy as np
from Coefficients_rwa_non_rwa import *
from MF_equations__with_A_squared import *
from numpy.linalg import inv
import time
import matplotlib.pyplot as plt
from pylab import figure, plot, show


# number of phonon states starting from | 0 > 
N = 4 

# Hamiltonian constants
eps = 1.0          # 2ls frequency
g_light_mat = 1.0  # light-mat coupling
S = 0.1            # exciton-phonon coupling
om_v = 0.2         # phonon freq

# number of molecules
N_m = 1.0

# Lindblad equation parameters
Gam_down = 1e-4  # excit decay
kappa = 1e-4     # photon leakage
Gam_z = 0.03     # exciton dephasing 
T = 0.025        # bath temp
gam_phon = 0.02  # phonon-bath coupling

# phot freq and pumping for phase diagram calculations
phot_freq = np.linspace(0.001, 2.2, 100)
Gam_up_array = np.linspace(0.0e-4, 1.6e-4, 100)



# defining GGM lambda matrices and structure factors f and g
from GGM import GGM_matr

lx = GGM_matr(N)[2]
lz = GGM_matr(N)[3]
lam = GGM_matr(N)[4]
f = GGM_matr(N)[5]
g = GGM_matr(N)[6]



# calculating coefficients A_i and B_i, which are independent on phot freq and incoherent rates
ec = exp_coef(N, eps, om_v, S, g_light_mat, g_light_mat, lam)
A = ec[0]
B = ec[1]


# creating array for a phase diagram (stable/unstable points)
ns_stable = np.zeros( ( len(Gam_up_array), len(phot_freq) ), dtype=int)
i = j = 0

start_time = time.time() 
for Gam_up in Gam_up_array:
    
    j = 0
        
    # Calculating rates gamma_i^{\mu}  (requires Lindblad parameters)
    gam_mu_i = rates(Gam_up, Gam_down, Gam_z, gam_phon, lam, N, T, om_v, S)

    # calculating zeta, xi and psi  (requires Hamiltonian and Lindblad parameters)
    zeta, xi, psi, phi, eta = equat_coeff(gam_mu_i, f, g, A, N)

    # normal state for stability matrix calculation
    A_vec = - (2j / N) * np.einsum('mj, mk, ijk -> i', gam_mu_i, np.conjugate(gam_mu_i), f)
    xi_inv_matr = inv(xi)
    lam_ns = np.real( np.dot(xi_inv_matr, A_vec) )

    
    for om_c in phot_freq:      

        # build stability matrix with A^2 term
        M = M_stability_with_A_squared(om_c, kappa, B, f, xi, lam_ns, N_m, g_light_mat, eps)
        
        # build stability matrix without A^2 term
        #M = M_stability(om_c, kappa, B, f, xi, lam_ns, N_m)

        # eigenvalues...
        max_eig, n_unstab = stable_eigs(M)

        ns_stable[i, j] = n_unstab

        j = j + 1
    i = i + 1
    print( i, "out of ", str(len(Gam_up_array)) )
        
print("--- %s seconds ---" % (time.time() - start_time))        






# plotting/saving results
ft = 14

Y, X = np.meshgrid(phot_freq, Gam_up_array / Gam_down)

plt.figure(figsize=(3.5, 2.5))
plt.contourf(Y, X, ns_stable, colors = ('white', 'black'), levels = [0,1,4])
plt.xlabel('$\omega_c$', size=ft)
plt.ylabel('$ \Gamma_{\\uparrow} / \Gamma_{\\downarrow} $', size=ft)
plt.title('$g \sqrt{N_m} = $' + str(g_light_mat), fontsize=ft, x = 0.75, y = 0.45)
plt.xticks(fontsize=ft, rotation=0)
plt.yticks(fontsize=ft, rotation=0)
#plt.text(0.5, 1.1, 'laser', fontsize = 16, color='white')
#plt.text(0.4, 0.74, 'normal', fontsize = 16, color='k')
#plt.text(0.1, 1.4, '(a)', fontsize = 18, color='w')

x = [0.0, 2.2]
y = [0.0, 1.6]
plt.xticks(np.arange(min(x), max(x)+0.1, 0.4))
plt.yticks(np.arange(min(y), max(y)+0.1, 0.4))



'''
plt.savefig('g_' + str(g_light_mat).replace(".","") + '__S_' + str(S).replace(".","") + 
                '__N_' + str(N) + '.svg', bbox_inches='tight', transparent=True)
'''

show(block=False)

