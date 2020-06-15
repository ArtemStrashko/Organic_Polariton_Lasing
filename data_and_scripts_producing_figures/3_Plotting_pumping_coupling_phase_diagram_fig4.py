
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, plot, show
import pickle
    
# for bare phot freq om_c = 1.0 choose a file Pump_coupl_ph_diagram__omc_10 
# for bare phot freq om_c = 1.2 choose a file Pump_coupl_ph_diagram__omc_12 
# for bare phot freq om_c = 0.8 choose a file Pump_coupl_ph_diagram__omc_0.8 
with open('Pump_coupl_ph_diagram__omc_10', 'rb') as handle:
    ph_diagr = pickle.load(handle)


# In[2]:


# plotting/saving results
ft = 14

fig, ax = plt.subplots(figsize=(3.5, 2.5))
plt.contourf(ph_diagr[0], ph_diagr[1], ph_diagr[2], colors = ('white', 'black'), levels = [0,1,4])
plt.xlabel('$g \sqrt{N_m}$', size=ft)
plt.ylabel('$ \Gamma_{\\uparrow} / \Gamma_{\\downarrow} $', size=ft)
plt.xticks(fontsize=ft, rotation=0)
plt.yticks(fontsize=ft, rotation=0)
#plt.text(0.05, 1.2, '$\omega_c = $' + str(om_c), fontsize = 18, color='w')
ax.set_xscale("log") 


#x = [0.0, 1.0]
#y = [0.0, 1.6]
#plt.xticks(np.arange(min(x), max(x)+0.1, 0.2))
#plt.yticks(np.arange(min(y), max(y)+0.1, 0.4))

#plt.savefig('phd_10_with_A2.svg', bbox_inches='tight', transparent=True)

show(block=False)


