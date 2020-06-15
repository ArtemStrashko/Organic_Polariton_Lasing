
# coding: utf-8

# In[6]:


import numpy as np
from pylab import figure, plot, show
import matplotlib.pyplot as plt

Gam_down = 1e-4  # excit decay


# In[7]:


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


# In[8]:


import pickle
    
with open('crit_pump__and__opt_freq', 'rb') as handle:
    g_lm_array, crit_pump, opt_freq = pickle.load(handle)


# In[9]:


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


# In[10]:


# critical pumping Gam_up vs coupling
# here each line corresponds to its optimal frequency, which may be read off from the figure above
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

