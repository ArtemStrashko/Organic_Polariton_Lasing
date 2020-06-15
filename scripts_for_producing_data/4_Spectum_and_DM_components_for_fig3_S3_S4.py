import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import time
from pylab import figure, plot, show

from Coefficients_rwa_non_rwa import *
from MF_equations__with_A_squared import *
from stability_for_gam_up_slices import *


# number of phonon states starting from | 0 > 
N = 4

# Hamiltonian constants
eps = 1.0          # 2ls freq
g_light_mat = 0.1  # light-mat coupling
S = 0.1            # exciton-phonon coupling
om_v = 0.2         # phon freq

# number of molecules
N_m = 1

# Lindblad equation parameters
Gam_down = 1e-4  # excit decay
kappa = 1e-4     # photon leakage
Gam_z = 0.03     # exciton dephasing 
T = 0.025        # phon bath temp
gam_phon = 0.02  # phonon-bath coupling

# array of photon freq
phot_freq = np.linspace(0.001, 2.0, 200)

# array of pumping Gamma_uparrow slices
Gam_up_array = np.array([0.4, 1.0, 1.8]) * 1e-4

# photon frequency with A^2 term vs bare phot freq
bare_phot_with_A2 = phot_freq * np.sqrt( 1 + (4 * g_light_mat**2 * N_m)/(eps * phot_freq) )











# defining GGM lambda matrices and structure factors f and g
from GGM import GGM_matr
lx = GGM_matr(N)[2]
lz = GGM_matr(N)[3]
lam = GGM_matr(N)[4]
f = GGM_matr(N)[5]
g = GGM_matr(N)[6]

# function for generating constant slices
def const_slice(Gam_up_array_for_slices, length_array, i):
    x = np.empty(len(length_array))
    x.fill(Gam_up_array_for_slices[i])
    return x

# calculating coefficients A_i and B_i  (requires Hamiltonian parameters)
ec = exp_coef(N, eps, om_v, S, g_light_mat, g_light_mat, lam)
A = ec[0]
B = ec[1]


# setting up zeros arrays
ns_stable = np.zeros( ( len(Gam_up_array), len(phot_freq) ), dtype=int)
all_eigenvalues_real_sort = np.zeros([len(Gam_up_array), len(phot_freq), np.shape(lam)[0] + 2 ], dtype = complex)
all_eigenvalues_imag_sort = np.zeros([len(Gam_up_array), len(phot_freq), np.shape(lam)[0] + 2 ], dtype = complex)
unstable_eigenvalues = np.zeros([len(Gam_up_array), len(phot_freq) ], dtype = complex)
all_phot_comp = np.zeros([len(Gam_up_array), len(phot_freq), np.shape(lam)[0] + 2])



i = j = 0
start_time = time.time() 
# calculating and sorting eigenvalues and eigenvectors
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
    
    for om_c in phot_freq:      

        # build stability matrix with A^2 term
        M = M_stability_with_A_squared(om_c, kappa, B, f, xi, lam_ns, N_m, g_light_mat, eps)
        
        # build stability matrix without A^2 term
        #M = M_stability(om_c, kappa, B, f, xi, lam_ns, N_m)

        # get eigenvalues and assosiated eigenvectors
        eigenValues, eigenVectors = linalg.eig(M)
        

        # sort the according to eigenvalues' real parts from smallest to largest
        idx_real_sort = eigenValues.argsort() 
        eigenValues_re_sort = eigenValues[idx_real_sort]            
        all_eigenvalues_real_sort[i,j,:] = eigenValues_re_sort
	# photon component of an eigenvector
        all_phot_comp[i,j,:] = abs(eigenVectors[0,idx_real_sort])**2 + abs(eigenVectors[1,idx_real_sort])**2
        
        
        # and now only unstable eigenvalue
        # sort the according to eigenvalues' real parts from smallest to largest and take only the largest one
        if np.real(eigenValues_re_sort[-1]) > 0:
            unstable_eigenvalues[i,j] = eigenValues_re_sort[-1]


        # eigenvalues...
        max_eig, n_unstab = stable_eigs(M)

        ns_stable[i, j] = n_unstab

        j = j + 1
    i = i + 1
    print( i, "out of ", str(len(Gam_up_array)) )
        
print("--- %s seconds ---" % (time.time() - start_time))        
        




# sorting imaginary eigenvalues
imag_eigenvalues_sort = np.zeros([np.shape(all_eigenvalues_real_sort)[0], 
                                  np.shape(all_eigenvalues_real_sort)[1], 
                                  np.shape(all_eigenvalues_real_sort)[2] ])

all_phot_comp_sort = np.zeros([np.shape(all_eigenvalues_real_sort)[0], 
                                  np.shape(all_eigenvalues_real_sort)[1], 
                                  np.shape(all_eigenvalues_real_sort)[2] ])

for i in range(len(all_eigenvalues_real_sort[:,1,2])):
    for j in range(len(all_eigenvalues_real_sort[1,:,2])): 
        imag_eigenvalues = np.imag(all_eigenvalues_real_sort[i,j,:])
        idx = imag_eigenvalues.argsort()
        imag_eigenvalues_sort[i,j,:] = imag_eigenvalues[idx] 
        all_phot_comp_sort[i,j,:] = all_phot_comp[i,j,idx]



# removing 00-01 transition line and also zero for clarity
for pum in range(len(Gam_up_array)):
    for om in range(len(phot_freq)-1):
        for num in range(np.shape(imag_eigenvalues_sort)[-1]):
            if ( ( abs(imag_eigenvalues_sort[pum, om, num] - imag_eigenvalues_sort[pum, om+1, num]) < 1e-8 and 
                imag_eigenvalues_sort[pum, om, num] > 0.2 and imag_eigenvalues_sort[pum, om, num]<0.4) or 
                (imag_eigenvalues_sort[pum, om, num] < 1e-8)):
                imag_eigenvalues_sort[pum, om, num] = np.nan










# choose the number of Gamma_up slice, for which you want to plot spectrum
i = 2 

# removing zero from imag of unstable part for clarity
unstab_im = np.array( np.abs( np.imag(unstable_eigenvalues[i,:])), dtype=np.double)
for ii in range(len(phot_freq)-1):
    if abs(unstab_im[ii] - unstab_im[ii+1]) > 0.5 or unstab_im[ii] < 1e-3:
        unstab_im[ii] = np.nan
        
# removing zero from re of unstable part for clarity
unstab_re = np.array( np.real(unstable_eigenvalues[i,:]), dtype=np.double)
for ii in range(len(phot_freq)-1):
    if abs(unstab_re[ii] - unstab_re[ii+1]) < 1e-10:
        unstab_re[ii+1] = np.nan







# plotting/saving results for the SPECTRUM
from matplotlib import rcParams
from matplotlib.collections import LineCollection

let = 'a' + 'b' + 'c'

ft = 18

x1 = phot_freq

y1 = imag_eigenvalues_sort[i,:,:]
y11 = unstab_im
yph = bare_phot_with_A2

x2 = phot_freq
y2 = unstab_re * 1e+2

fig, ax1 = plt.subplots(figsize=(3.5, 2.7*1.2))

plt.yticks(fontsize = ft, rotation=0)
plt.xticks(fontsize = ft, rotation=0)

y = [0.0, 2.1]
plt.yticks(np.arange(min(y), max(y)+0.1, 0.5), ())

plt.text(0.03, 1.8, '(' + let[i] + ')', fontsize = ft+2, color='k')
#plt.text(0.52, 2.3, '$g \sqrt{N_m} = $' + str(g_light_mat), fontsize=ft)
ax2 = ax1.twinx()
ax1.set_xlim(0, 2) 
ax1.plot(x1, bare_phot_with_A2, '--', dashes=(2, 6), color = 'hotpink', linewidth = 2.8)#, label = 'bare photon')


for jjj in range(y1.shape[1]):
    col = all_phot_comp_sort[i,:,jjj]**(0.2)
    t=np.copy(col)
    t[0]=max(col[0],col[1])
    t[-1]=max(col[-1],col[-2])
    for k in range(1,len(t)-1):
        t[k] = max(max(col[k],col[k-1]),col[k+1])

    # set up a list of (x,y) points
    points = np.array([x1,y1[:,jjj]]).transpose().reshape(-1,1,2)

    # set up a list of segments
    segs = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)

    # make the collection of segments
    lc = LineCollection(segs, cmap=plt.get_cmap('Greys'), linewidth = 2.5)
    lc.set_clim([0,1])
    lc.set_array(t) # color the segments by our parameter
    ax1.add_collection(lc)
    
    
ax1.plot(x1, y11, '--', dashes = (4,6), linewidth = 3.5, color = 'y' )
ax1.set_ylim(0, 2) 
ax2.plot(x2, y2, color = 'dodgerblue')
ax2.set_ylim(ymin = 0) 
plt.yticks(fontsize = ft, rotation=0)
#ax1.set_xticklabels([])

x = [0.0, 2.1]
plt.xticks(np.arange(min(x), max(x)+0.1, 0.5),() )
y = [0.0, 2.41]
plt.yticks(np.arange(min(y), max(y), 0.6), color = 'dodgerblue')

plt.tick_params(axis='both', which='major', labelsize = ft)
plt.rcParams['font.size'] = ft


ax1.set_xlabel('$\omega_c$', fontsize = ft+4)
ax1.set_ylabel('freq, $Im \\xi$', fontsize = ft)
ax2.set_ylabel('gain, $Re \\xi \\times 10^{-2}$', color = 'dodgerblue', fontsize = ft)


plt.title('$\Gamma_{\\uparrow} = $' + str(np.round(Gam_up_array[i] / Gam_down, 1)) + '$\Gamma_{\\downarrow}$', 
          fontsize=ft, x = 0.52, y = 1)
        
'''
plt.savefig('spectrum_and_gain_coupl' + str(g_light_mat).replace(".","") + 
            '__pump' + str(np.round(Gam_up_array[i] / Gam_down, 1)).replace(".","") + 
            '.pdf', bbox_inches='tight', transparent=True)
'''
plt.show()














## Density Matrix components of an unstable mode

# calculating density matrix components for plotting molecular transitions involved in lasing
# first argument True means with A^2 term, false - without A^2 term
rho_tot = stab_for_gamup_slices(True, N, eps, om_v, S, g_light_mat, g_light_mat, lam, 
                                                     phot_freq, Gam_up_array, Gam_down, 
                                                     Gam_z, gam_phon, T, f, g, kappa, N_m)


# choose the number of Gamma_up slice
nn = 2 

# extracting molecular transitions' weights
rho_tot_for_plots = abs( rho_tot[nn, :, 0:int(N), int(N):] ) + abs( rho_tot[nn, :, int(N):, 0:int(N)] )

# plotting/saving the results
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.plot(phot_freq,  rho_tot_for_plots[:, 0, 0],  label = str(0) + "-" + str(0), color = 'k' )
ax.plot(phot_freq,  rho_tot_for_plots[:, 1, 0],  label = str(1) + "-" + str(0), color = 'r' )
ax.plot(phot_freq,  rho_tot_for_plots[:, 0, 1],  label = str(0) + "-" + str(1), color = 'b' )

lett = 'd' + 'e' + 'f'

x = [0.0, 2.0]
plt.xticks(np.arange(min(x), max(x)+0.1, 0.5))

y = [0.0, 0.31]
plt.yticks(np.arange(min(y), max(y)+0.1, 0.1),())
plt.xlabel('$\omega_c$', size=ft+4)
#plt.legend(loc = [0.6, 0.2], fontsize = ft-2, handlelength = 1)
ax.set_ylabel('weight', fontsize = ft)
plt.xlim(0.0, 2.0)
plt.ylim(0, 0.3)

plt.text(0.03, 0.26, '(' + lett[nn] + ')', fontsize = ft+2, color='k')

'''
plt.savefig('DM_comp_coupl' + str(g_light_mat).replace(".","") + 
            '__pump' + str(np.round(Gam_up_array[nn] / Gam_down, 1)).replace(".","") + 
            '.pdf', bbox_inches='tight', transparent=True)
'''

show(block=False)

