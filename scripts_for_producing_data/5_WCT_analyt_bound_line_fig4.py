# ## Input parameters

# emision/absorption spectrum calculations
import numpy as np
from WCT_equations import *  
import pylab as plt
import time
from pylab import figure, plot, show
import matplotlib.pyplot as plt

# number of phonon states starting from | 0 > 
N = 4

# Hamiltonian constants
om_c = 0.8    # photon freq
eps = 1.0     # excit freq
S = 0.1 # exciton-phonon coupling
om_v = 0.2    # phon freq
g_light_mat_array = np.logspace(-3, 1e-0, 100) * 1e-1

# number of molecules (any for mean-field, but large enough for correct cumulant results, e.g. N_m >= 100)
N_m = 1e+2


# Lindblad equation parameters
Gam_down = 1e-4  # excit decay
Gam_up_array = np.linspace(0.0, 4e-4, 100)
kappa = 1e-4     # photon leakage
Gam_z = 0.03     # exciton dephasing 
T = 0.025    # phon bath temp
gam_phon = 0.02 # phonon-bath coupling


# emission, absorption spectra and range of frequencies
start_time = time.time()
em_abs_freq = emis_absorp_freq(N, eps, om_v, S, gam_phon, Gam_z, T)
print("--- %s seconds ---" % (time.time() - start_time))   

em = em_abs_freq[0]   # emission spectrum
ab = em_abs_freq[1]   # absorption spectrum

freq = em_abs_freq[2] # frequency array

# searching for index corresponding to the photon freq we need
ind_corresp_to_phot_freq = index_of_element(freq, om_c)

# check that this is correct
freq[ind_corresp_to_phot_freq]

# setting up empty arrays for mean field and cumulant calculations of threshold pumping 
gam_th_second_ord_cum = np.empty([len(g_light_mat_array)])
gam_th_mean_field     = np.empty([len(g_light_mat_array)])

# calculating threshold pumping vs coupling
i = 0    
for g_light_mat in g_light_mat_array:

    Emis = g_light_mat**2 * em[ind_corresp_to_phot_freq]
    Abs =  g_light_mat**2 * ab[ind_corresp_to_phot_freq]

    # cumulant calc
    gam_th_second_ord_cum[i] = (kappa*(Gam_down + Emis) + N_m * Abs * Gam_down ) / (N_m * Emis - kappa)

    # mean-field calc
    gam_th_mean_field[i] = ( (N_m * Abs + kappa) / (N_m * Emis - kappa) ) * Gam_down
    
    i = i + 1


# plotting/saving results
fig, ax = plt.subplots(1, 1)
#ax.semilogx(g_light_mat_array*np.sqrt(N_m), gam_th_second_ord_cum/Gam_down, linewidth = 2.0)
ax.semilogx(g_light_mat_array*np.sqrt(N_m), gam_th_mean_field/Gam_down, '--', linewidth = 2.0)
plt.xlabel('$g \sqrt{N_m}$', size=20)
ax.set_ylim(0.0, 2.0)

# plt.savefig('phot_pyth.png')

plt.show()

