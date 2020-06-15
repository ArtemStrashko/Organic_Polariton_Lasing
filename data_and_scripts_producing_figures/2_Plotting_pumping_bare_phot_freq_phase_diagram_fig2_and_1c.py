
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, plot, show
import pickle
    
# for coupling g \sqrt{N_m} = 0.05 choose a file Pum_freq_ph_diagram__coupling_005
# for coupling g \sqrt{N_m} = 0.1 choose a file Pum_freq_ph_diagram__coupling_01 
# for coupling g \sqrt{N_m} = 0.4 choose a file Pum_freq_ph_diagram__coupling_04  
# for coupling g \sqrt{N_m} = 0.7 choose a file Pum_freq_ph_diagram__coupling_07
# for coupling g \sqrt{N_m} = 1.0 choose a file Pum_freq_ph_diagram__coupling_10
with open('Pum_freq_ph_diagram__coupling_005', 'rb') as handle:
    phase_diagram = pickle.load(handle)


# In[3]:


ft = 14

plt.figure(figsize=(3.5, 2.5))
plt.contourf(phase_diagram[0], phase_diagram[1], phase_diagram[2], colors = ('white', 'black'), levels = [0,1,4])
plt.xlabel('$\omega_c$', size=ft)
plt.ylabel('$ \Gamma_{\\uparrow} / \Gamma_{\\downarrow} $', size=ft)
#plt.title('$g \sqrt{N_m} = $' + str(g_light_mat), fontsize=ft, x = 0.75, y = 0.45)
plt.xticks(fontsize=ft, rotation=0)
plt.yticks(fontsize=ft, rotation=0)
#plt.text(0.5, 1.1, 'laser', fontsize = 16, color='white')
#plt.text(0.4, 0.74, 'normal', fontsize = 16, color='k')
#plt.text(0.1, 1.4, '(a)', fontsize = 18, color='w')

x = [0.0, 2.2]
y = [0.0, 4]
plt.xticks(np.arange(min(x), max(x)+0.1, 0.4))
plt.yticks(np.arange(min(y), max(y)+0.1, 1.0))



'''
plt.savefig('g_' + str(g_light_mat).replace(".","") + '__S_' + str(S).replace(".","") + 
                '__N_' + str(N) + '.svg', bbox_inches='tight', transparent=True)
'''

show(block=False)

