## Setting up parameters, importing functions

import numpy as np
from Coefficients_rwa_non_rwa import *
from MF_equations__with_A_squared import *

import time
from scipy.integrate import ode
from pylab import figure, plot, show


import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from numpy.linalg import inv

# number of phonon states starting from | 0 > 
N = 4 

# Hamiltonian constants
eps = 1.0     # excit freq
S = 0.1       # exciton-phonon coupling
om_v = 0.2    # phon freq

# number of molecules
N_m = 1.0

# Lindblad equation parameters
Gam_down = 1e-4  # excit decay
kappa = 1e-4     # photon leakage
Gam_z = 0.03     # exciton dephasing 
T = 0.025    # phon bath temp
gam_phon = 0.02 # phonon-bath coupling



# defining GGM lambda matrices and structure factors f and g
from GGM import GGM_matr

lx = GGM_matr(N)[2]
lz = GGM_matr(N)[3]
lam = GGM_matr(N)[4]
f = GGM_matr(N)[5]
g = GGM_matr(N)[6]

# precalculations
zeta = 1j * f + g
prec = np.einsum("ijl, klp -> ijkp", f, zeta)
precc = np.einsum("kil, ljp -> kijp", f, zeta)


# stability matrix
def stable_eigs(M):
    
    from numpy.linalg import eigvals
    from numpy import amax
    
    stabmat = M
    evals = eigvals(stabmat)
   
    n_unstab =0
    for count in range(len(evals)):
        if evals[count].real>0.0:
            n_unstab+=1
    
    return n_unstab 



# calculating coefficients A_i and B_i  (requires Hamiltonian parameters)
ec = exp_coef(N, eps, om_v, S, 1.0, 1.0, lam)
A = ec[0] # doesn't depend on matter-light coupling
BB = ec[1] # proportional to m-l coupling, so later will be updated B = g sqrt{N_m} * B
prec_1 = np.einsum("ijp, j -> ip", f, A) # just useful precalculation









## defining functions for rough minima and bounds finding

# normal state precalculations (densiti matrix and xi tensor in the normal state vs pumping Gam_up)
def normal_state(Gam_up_array):

    xi_pumping_array = np.zeros([np.shape(lam)[0],np.shape(lam)[0],len(Gam_up_array)], dtype = complex)
    lam_ns_pumping_array = np.zeros([np.shape(lam)[0],len(Gam_up_array)], dtype = complex)
    jj = 0
    for Gam_up in Gam_up_array:

        j = 0

        # Calculating rates gamma_i^{\mu}  (requires Lindblad parameters)
        gam_mu_i = rates(Gam_up, Gam_down, Gam_z, gam_phon, lam, N, T, om_v, S)
        gam_sum = np.einsum('mj, mk -> jk', gam_mu_i, np.conjugate(gam_mu_i))

        # calculating zeta, xi and psi  (requires Hamiltonian and Lindblad parameters)
        xi_pumping_array[:,:,jj] = (1j * ( np.einsum("jk, ijkp ->ip", gam_sum, prec) + 

                    np.einsum("jk, kijp ->ip", gam_sum, precc) 
                   ) 
                      +  2 * prec_1  

                )

        # some auxiliary functions for stability matrix calculation
        A_vec = - (2j / N) * np.einsum('jk, ijk -> i', gam_sum, f)
        xi_inv_matr = inv(xi_pumping_array[:,:,jj])
        lam_ns_pumping_array[:,jj] = np.real( np.dot(xi_inv_matr, A_vec) )

        jj = jj + 1
        
    return xi_pumping_array, lam_ns_pumping_array


# calculating lower pumping boundary
def gam_up_low_boundary(xi_pumping_array, lam_ns_pumping_array, coupling_lm, BB_term):

    B = BB_term * coupling_lm
    # lower boundary calculations
    gam_up_position = np.zeros([len(phot_freq)])

    # stability calculations
    i = 0
    for om_c in phot_freq:

        j = 0
        for Gam_up in Gam_up_array:

            xi = xi_pumping_array[:,:,j]
            lam_ns = lam_ns_pumping_array[:,j]

            # stability matrix 
            M = M_stability_with_A_squared(om_c, kappa, B, f, xi, lam_ns, N_m, coupling_lm, eps)

            # eigenvalues
            n_unstab = stable_eigs(M)

            if n_unstab > 0:
                gam_up_position[i] = Gam_up
                break
            else:
                pass

            j = j + 1

        i = i + 1

        if (om_c > 1.0 and gam_up_position[i-1] == 0):
            break

    return gam_up_position


def find_all_loc_min(pump_array):
    # this functions finds all local minima
    loc_minima = []
    flag = True
    # interating over all the points 
    for i in range(len(pump_array)):
        if pump_array[i] == 0.0:
            continue
        # if next value is higher than previous and slope before was negative, max is found  
        if i > 0 and (pump_array[i-1] < pump_array[i]) and flag and pump_array[i-1] != 0:
            loc_minima.append(i-1)
            flag = False
        # if slope is positive should not check 'if' above
        if i > 1 and (pump_array[i-1] > pump_array[i]) and not flag:
            flag = True
    loc_minima = np.array(loc_minima)
        
    # returns indices of freq array corresponding to loc min pumping 
    return loc_minima


def find_freq_boundaries(pump_array, phot_freq, loc_min):
    boundaries = []
    # finding left and right boundary of local minima fur further more accurate calculations
    for loc_min_position in loc_min:
        # start from the position of local minima
        lb = loc_min_position
        rb = loc_min_position
        # go left until pump is higher
        while pump_array[loc_min_position] >= pump_array[lb]:
            lb = lb - 1
        boundaries.append(phot_freq[lb-1])
        # go right until pump is higher
        while pump_array[loc_min_position] >= pump_array[rb]:
            rb = rb + 1
        boundaries.append(phot_freq[rb+1])
        
    boundaries = np.array(boundaries)
    # returns an array of left and right bundaries for every loc minimum
    return np.array(boundaries)










## functions for freq optimization

def is_stable(g_l_m, freq, pump):
    
    # Calculating rates gamma_i^{\mu}  (requires Lindblad parameters)
    gam_mu_i = rates(pump, Gam_down, Gam_z, gam_phon, lam, N, T, om_v, S)
    gam_sum = np.einsum('mj, mk -> jk', gam_mu_i, np.conjugate(gam_mu_i))

    # calculating zeta, xi and psi  (requires Hamiltonian and Lindblad parameters)
    xi = (1j * ( np.einsum("jk, ijkp ->ip", gam_sum, prec) + 

                np.einsum("jk, kijp ->ip", gam_sum, precc) 
               ) 
                  +  2 * prec_1  

            )

    # normal state calculation
    A_vec = - (2j / N) * np.einsum('jk, ijk -> i', gam_sum, f)
    xi_inv_matr = inv(xi)
    lam_ns = np.real( np.dot(xi_inv_matr, A_vec) )
    
    # stability matrix
    M = M_stability_with_A_squared(freq, kappa, BB * g_l_m, f, xi, lam_ns, N_m, g_l_m, eps)
    
    if stable_eigs(M) > 0:
        stable = False
    else:
        stable = True
        
    return stable



def get_crit_pump(lm_coupl, phot_freq, min_pump, max_pump, epsilon):
    
    while is_stable(lm_coupl, phot_freq, max_pump):
        print('Can you imagine - I need to increase pumping!')
        max_pump = 1.5 * max_pump
        if max_pump > Gam_down * 2e+2:
            G_2 = Gam_down * 2e+2
            break

    if max_pump < Gam_down * 2e+2:
        
        G_1 = min_pump
        G_3 = max_pump
        G_2 = 0.5 * (min_pump + max_pump)

        while abs(G_1 - G_3) / G_2 > epsilon:
            if is_stable(lm_coupl, phot_freq, G_2):
                G_1 = G_2
                G_3 = G_3
                G_2 = 0.5 * (G_1 + G_3)
            else:
                G_1 = G_1
                G_3 = G_2
                G_2 = 0.5 * (G_1 + G_3)
            
    return G_2



# light-mat coupling array
g_lm_array = np.exp( np.linspace(np.log(0.0024), np.log(10), 300) )








# accuracy of crit pump finding
epsilon = 0.001
depth = 50

flag1, flag2, flag3, flag4, flag5, flag6, flag7 = True, False, False, False, False, False, False


crit_pump = np.zeros([4, len(g_lm_array)]) # supposing that there are four local minima
opt_freq = np.zeros([4, len(g_lm_array)])
flag = False


start_time = time.time()  
# iterating over coupling from large to small
for i in range(len(g_lm_array)):
    g_light_mat = g_lm_array[i]
    
    print('i = ' + str(i+1) + ' out of ' + str(len(g_lm_array)))    
    print(  'coupling is ' + str( np.round(g_light_mat,4) )  )
    
    # normal state precalculations
    if g_light_mat <= 0.003 and flag1:
        phot_freq = np.linspace(0.95, 1.05, 300)
        Gam_up_array = np.linspace(3.0, 30.0, 400) * Gam_down
        print('calculate normal state, coupling is ' + str(g_light_mat) )
        xi_pumping_array, lam_ns_pumping_array = normal_state(Gam_up_array)
        flag1 = False
        flag2 = True
    if 0.003 < g_light_mat <= 0.005 and flag2:
        phot_freq = np.linspace(0.75, 1.1, 300)
        Gam_up_array = np.linspace(0.2, 20.0, 400) * Gam_down
        print('calculate normal state, coupling is ' + str(g_light_mat) )
        xi_pumping_array, lam_ns_pumping_array = normal_state(Gam_up_array)
        flag2 = False
        flag3 = True
    if 0.005 < g_light_mat <= 0.06 and flag3:
        phot_freq = np.linspace(0.1, 1.0, 300)
        Gam_up_array = np.linspace(0.2, 1.6, 400) * Gam_down
        print('calculate normal state, coupling is ' + str(g_light_mat) )
        xi_pumping_array, lam_ns_pumping_array = normal_state(Gam_up_array)
        flag3 = False
        flag4 = True
    if 0.07 < g_light_mat <= 0.1 and flag4:
        phot_freq = np.linspace(0.1, 1.0, 600)
        Gam_up_array = np.linspace(0.2, 1.6, 800) * Gam_down
        print('calculate normal state, coupling is ' + str(g_light_mat) )
        xi_pumping_array, lam_ns_pumping_array = normal_state(Gam_up_array)
        flag4 = False
        flag5 = True
    if 0.1 < g_light_mat <= 0.3 and flag5:
        phot_freq = np.linspace(0.1, 0.9, 300)
        Gam_up_array = np.linspace(0.2, 0.5, 400) * Gam_down
        print('calculate normal state, coupling is ' + str(g_light_mat) )
        xi_pumping_array, lam_ns_pumping_array = normal_state(Gam_up_array)
        flag5 = False
        flag6 = True
    if 0.3 < g_light_mat <= 1.0 and flag6:
        phot_freq = np.exp( np.linspace(np.log(0.01), np.log(1.5), 300) )
        Gam_up_array = np.linspace(0.2, 0.44, 400) * Gam_down
        print('calculate normal state, coupling is ' + str(g_light_mat) )
        xi_pumping_array, lam_ns_pumping_array = normal_state(Gam_up_array)
        flag6 = False
        flag7 = True
    if 1.0 < g_light_mat and flag7:
        phot_freq = np.exp( np.linspace(np.log(0.0002), np.log(100), 300) )
        Gam_up_array = np.linspace(0.2, 0.385, 400) * Gam_down
        print('calculate normal state, coupling is ' + str(g_light_mat) )
        xi_pumping_array, lam_ns_pumping_array = normal_state(Gam_up_array)
        flag7 = False
        
    # approximately finding local minima and bounds for accurate optimization around these minima
    gam_up_position = gam_up_low_boundary(xi_pumping_array, lam_ns_pumping_array, g_light_mat, BB)
    loc_min = find_all_loc_min(gam_up_position)
    #loc_min_freq = phot_freq[ find_all_loc_min(gam_up_position) ]
    freq_bound = find_freq_boundaries(gam_up_position, phot_freq, loc_min)
    
    num_of_minima = len(loc_min)
    print('  number of local minima is ' + str(num_of_minima))         
    
    min_pump = Gam_up_array[0]
    max_pump = Gam_up_array[-1]
        
    # optimization for each minimum
    for kk in range(num_of_minima):
        # starting from the highest freq
        wmax = freq_bound[-2*kk - 2]
        wmin = freq_bound[-2*kk - 1]
        
        print('kk = ' + str(kk))
        print('wmax = ' + str(wmax))
        print('wmin = ' + str(wmin))
    
        # Find the critical pump at these frequencies
        fmin = get_crit_pump(g_light_mat, wmin, min_pump, max_pump, epsilon)
        fmax = get_crit_pump(g_light_mat, wmax, min_pump, max_pump, epsilon)

        # Repeat depth times, for accuracy 2^depth
        for j in range(depth):
            wnew = 0.5 * (wmin + wmax)
            fnew = get_crit_pump(g_light_mat, wnew, min_pump, max_pump, epsilon)

            # Check this was actually lower
            if (fnew > 0.999*min(fmin,fmax)) and abs((fmin-fmax)/fmax) < 0.001:
                
                if kk == 0 and (not flag) and 0.9 > wnew > 0.7:
                    flag = True
                
                if not flag:
                    # lasing, so append freq and pump
                    crit_pump[-kk, i] = fnew
                    opt_freq[-kk, i]  = wnew
                    print('  found minimum, number of bisections = ' + str(j))
                    print('  pump = ' + str(fnew/Gam_down) + ' , freq = ' + str(wnew) )
                    break
                else:
                    # lasing, so append freq and pump
                    crit_pump[-(kk+1), i] = fnew
                    opt_freq[-(kk+1), i]  = wnew
                    print('  found minimum, number of bisections = ' + str(j))
                    print('  pump = ' + str(fnew/Gam_down) + ' , freq = ' + str(wnew) )
                    break


            # Work out which end point to replace.
            if (fmin<fmax):
                # The new range should be wmin to wnew
                wmax = wnew
                fmax = fnew
            else:
                # The new range should be wnew to wmax
                wmin = wnew
                fmin = fnew

            if j == depth - 1:
                crit_pump[-kk, i] = fnew
                opt_freq[-kk, i]  = wnew
                print('  quit loop, j = ' + str(j))
                print('  pump = ' + str(fnew/Gam_down) + ' , freq = ' + str(wnew) )
    print('\n')

print("--- %s seconds ---" % (time.time() - start_time) )   



















## Plotting



ft = 14

# critical pumping Gam_up vs coupling
plt.figure(figsize=(5, 4))
plt.semilogx(g_lm_array, crit_pump[0,:] / Gam_down )
plt.semilogx(g_lm_array, crit_pump[1,:] / Gam_down )
plt.semilogx(g_lm_array, crit_pump[2,:] / Gam_down )
plt.semilogx(g_lm_array, crit_pump[3,:] / Gam_down )
plt.xlabel('$g \sqrt{N_m}$', size=ft)
plt.ylabel('$ \Gamma_{\\uparrow} / \Gamma_{\\downarrow} $', size=ft)
plt.xticks(fontsize=ft, rotation=0)
plt.yticks(fontsize=ft, rotation=0)
plt.ylim(0.0, 4.0)
#plt.xlim(0.005, 0.1)

show(block=False)


# optimal frequency vs coupling
plt.figure(figsize=(5, 4))
plt.semilogx(g_lm_array, opt_freq[0,:] )
plt.semilogx(g_lm_array, opt_freq[1,:] )
plt.semilogx(g_lm_array, opt_freq[2,:] )
plt.semilogx(g_lm_array, opt_freq[3,:] )
plt.xlabel('$g \sqrt{N_m}$', size=ft)
plt.ylabel('opt freq', size=ft)
#plt.legend()
plt.xticks(fontsize=ft, rotation=0)
plt.yticks(fontsize=ft, rotation=0)
plt.ylim(0.0, 2.0)
#plt.xlim(0.005, 0.1)

show(block=False)





def remove_zeros(coupl, freq):
    new_freq = []
    new_coupl = []
    for i in range(len(coupl)-1):
        if freq[i] != 0:
            new_freq.append(freq[i])
            new_coupl.append(coupl[i])
    new_freq = np.array(new_freq)
    new_coupl = np.array(new_coupl)
    return new_coupl, new_freq


# In[85]:


# optimal bare phot freq vs coupling
ft = 16
plt.figure(figsize=(5,4*0.8))
plt.semilogx(remove_zeros(g_lm_array[2:19],opt_freq[0,2:19])[0], 
             remove_zeros(g_lm_array[2:19], opt_freq[0,2:19])[1], color = 'grey', linewidth = 5)
plt.semilogx(remove_zeros(g_lm_array,opt_freq[0,:])[0], remove_zeros(g_lm_array, opt_freq[0,:])[1] , color = 'k')
plt.semilogx(remove_zeros(g_lm_array,opt_freq[1,:])[0], remove_zeros(g_lm_array,opt_freq[1,:])[1], color = 'k')
plt.semilogx(remove_zeros(g_lm_array,opt_freq[2,:])[0], remove_zeros(g_lm_array,opt_freq[2,:])[1], color = 'k')
plt.semilogx(remove_zeros(g_lm_array[19:],opt_freq[3,19:])[0], 
             remove_zeros(g_lm_array[19:],opt_freq[3,19:])[1], color = 'grey', linewidth = 5)
plt.semilogx(remove_zeros(g_lm_array,opt_freq[3,:])[0], remove_zeros(g_lm_array,opt_freq[3,:])[1], color = 'k')
#plt.xlabel('$g \sqrt{N_m}$', size=ft)
plt.ylabel('optimal $\omega_c$', size=ft)
plt.text(0.002, 1.1, '(0-0)', fontsize = ft-2)
plt.text(0.03, 0.85, '(1-0)', fontsize = ft-2)
plt.text(0.1, 0.62, '(2-0)', fontsize = ft-2)
plt.text(0.1, 0.37, '(3-0)', fontsize = ft-2)

plt.xticks(fontsize=ft, rotation=0)
plt.yticks(fontsize=ft, rotation=0)
plt.ylim(0.0, 1.6)
plt.xlim(0.001, 10.0)
#plt.savefig('opt_freq.svg', bbox_inches='tight', transparent=True)

show(block=False)

