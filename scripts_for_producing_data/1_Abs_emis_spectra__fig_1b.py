# $H = \varepsilon \sigma^z_{ex} + \Omega b^{\dagger} b + \Omega \sqrt{S} \sigma^z_{ex} (b^{\dagger} + b)$

# Qutip, exact results


import numpy as np
from qutip import *
import pylab as plt

# number of phonon states { |0>, |1>, ... }
N = 4

# Hamiltonian parameters
eps = 1.0  # 2ls energy
om_v = 0.2 # molecular vibrtion frequency
S = 0.1    # exciton=phonon coupling

# Dissipation parameters
gam_phon = 0.02 # phon-bath coupling constant
T = 0.025       # bath temperature
n = 1.0 / (np.exp(om_v / T) - 1.0)
Gam_z = 0.03    # exciton dephasing

# System operators
S_z_ex  = tensor(sigmaz(), identity(N))
S_m_ex  = tensor(sigmam(), identity(N))
S_pm_ex = S_m_ex.dag() * S_m_ex
b = tensor(identity(2), destroy(N))

# Hamiltonian
H_N = 0.5 * eps * S_z_ex + om_v * ( b.dag() * b  + np.sqrt(S) * S_z_ex * (b.dag() + b) )


tlist = np.linspace(0, 1000, 100000)

# jump operators for Lindblad equation
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



# calculating emission spectrum
Gam_up = 1e-6 # infinitesimal excit pumping
Gam_down = 0.0 * Gam_up  # zero excit decay
corr_emission = correlation_2op_1t(H_N, None, tlist, c_ops_N(), S_m_ex.dag(), S_m_ex)
wlist1, spec1 = spectrum_correlation_fft(tlist, corr_emission)

# absorption spectrum
Gam_down = 1e-6 # infinitesimal excit decay
Gam_up = 0.0 * Gam_down  # zero excit pumping
corr_absorption = correlation_2op_1t(H_N, None, tlist, c_ops_N(), S_m_ex, S_m_ex.dag())
corr_absorption=np.conj(corr_absorption)
wlist2, spec2 = spectrum_correlation_fft(tlist, corr_absorption)


# emission and absorption spectra
emission = spec1
absorption = spec2


# plot the spectra
ft = 14
plt.figure(figsize=(3.5, 2.5))
plt.plot(wlist2 , spec2 / max(spec2), label = 'abs')
plt.plot(wlist1 , spec1 / max(spec1), label = 'emiss')
plt.xlim(0.0, 2.0)
plt.ylim(0, 1)
plt.legend(loc = [0.6, 0.62], fontsize=14, handlelength = 1)
plt.xlabel('$\omega_c$', size=ft)
plt.xticks(fontsize = ft, rotation=0)
plt.yticks(fontsize = ft, rotation=0)
plt.text(0.22, 0.83, '(b)', fontsize = 18, color='k')

x = [0.0, 2.0]
y = [0.0, 1.0]
plt.xticks(np.arange(min(x), max(x)+0.1, 0.4))
plt.yticks(np.arange(min(y), max(y)+0.1, 0.2))

'''
plt.savefig('abs_emis__S' + str(S).replace(".","") + 
                '__N_' + str(N) + '.svg', bbox_inches='tight', transparent=True)
'''                

plt.show(block=False)

