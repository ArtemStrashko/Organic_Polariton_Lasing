
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
from pylab import figure, plot, show


# In[5]:


with open('molec_weights_at_the_boundary', 'rb') as handle:
    phot_freq_stab, rho_tot_for_plots = pickle.load(handle)


# In[6]:


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
#ax.yaxis.offsetText.set_fontsize(ft)

x = [0.0, 2.0]
y = [0.0, 0.16]
plt.xticks(np.arange(min(x), max(x)+0.1, 0.4))
plt.yticks(np.arange(min(y), max(y), 0.05))

#plt.text(0.1, 0.1, '(d)', fontsize = 18, color='k')

#plt.savefig('DM_components.pdf', bbox_inches='tight', transparent=True)

show(block=False)

