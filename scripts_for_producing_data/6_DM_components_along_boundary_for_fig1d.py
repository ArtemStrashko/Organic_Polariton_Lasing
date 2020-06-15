import numpy as np
from Coefficients_rwa_non_rwa import *
from MF_equations__with_A_squared import *
from stability_along_boundary import *  

import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
from pylab import figure, plot, show


# number of phonon states starting from | 0 > 
N = 4 

# Hamiltonian constants
eps = 1.0           # excit freq
g_light_mat = 0.05  # light-mat coupling
S = 0.1             # exciton-phonon coupling
om_v = 0.2          # phon freq

# number of molecules
N_m = 1.0

# Lindblad equation parameters
Gam_down = 1e-4  # excit decay
kappa = 1e-4     # photon leakage
Gam_z = 0.03     # exciton dephasing 
T = 0.025        # bath temp
gam_phon = 0.02  # phonon-bath coupling

phot_freq_stab = np.linspace(0.001, 2.2, 1000)
Gam_up_array_stab = np.linspace(0.0e-4, 6e-4, 1000)




# defining GGM lambda matrices and structure factors f and g
from GGM import GGM_matr
lx = GGM_matr(N)[2]
lz = GGM_matr(N)[3]
lam = GGM_matr(N)[4]
f = GGM_matr(N)[5]
g = GGM_matr(N)[6]

# costant array 
def const_slice(Gam_up_array_for_slices, length_array, i):
    x = np.empty(len(length_array))
    x.fill(Gam_up_array_for_slices[i])
    return x


# stability at the normal-lasing boundary (density matrix components and position of pumping Gam_up for every photon frequency)
rho_tot, gam_up_position = unstab_bound_vertical_calc(N, eps, om_v, S, g_light_mat, g_light_mat, 
                                                                  lam, phot_freq_stab, Gam_up_array_stab, 
                                                                  Gam_down, Gam_z, gam_phon, T, f, g, kappa, N_m)   

# DM components for plotting molecular transitions weights
rho_tot_for_plots = abs( rho_tot[:, 0:int(N), int(N):] ) + abs( rho_tot[:, int(N):, 0:int(N)] )









# plotting/saving results
ft = 14

plt.figure(figsize=(3.5, 2.5))

'''
# for plotting all the components
for i in range(N):
    for j in range(N):
        plt.plot(phot_freq_stab,  rho_tot_for_plots[:, i, j], label = str(i) + "-" + str(j) )
'''
# plotting only 0-0, 1-0, 0-1 transitions
plt.plot(phot_freq_stab,  rho_tot_for_plots[:, 0,0], label = str(0) + "-" + str(0), color = 'k' )
plt.plot(phot_freq_stab,  rho_tot_for_plots[:, 1,0], label = str(1) + "-" + str(0), color = 'r' )
plt.plot(phot_freq_stab,  rho_tot_for_plots[:, 0,1], label = str(0) + "-" + str(1), color = 'b' )

plt.xticks(fontsize=ft, rotation=0)
plt.yticks(fontsize=ft, rotation=0)
plt.xlabel('$\omega_c$', size=ft)
plt.ylim(0, 0.14)
plt.xlim(0,2)

plt.legend(loc = [0.71, 0.5], fontsize=ft-1, handlelength = 1)
ax.yaxis.offsetText.set_fontsize(ft)

x = [0.0, 2.0]
y = [0.0, 0.16]
plt.xticks(np.arange(min(x), max(x)+0.1, 0.4))
plt.yticks(np.arange(min(y), max(y), 0.05))

plt.text(0.1, 0.1, '(d)', fontsize = 18, color='k')

#plt.savefig('DM_components.pdf', bbox_inches='tight', transparent=True)

show(block=False)




