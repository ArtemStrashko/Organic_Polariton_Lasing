
# coding: utf-8

# In[1]:


import pylab as plt
from pylab import figure, plot, show
import matplotlib.pyplot as plt


# In[2]:


import pickle

with open('WCT_border_line__08', 'rb') as handle:
    border_08 = pickle.load(handle)
    
with open('WCT_border_line__10', 'rb') as handle:
    border_10 = pickle.load(handle)
    
with open('WCT_border_line__12', 'rb') as handle:
    border_12 = pickle.load(handle)


# In[3]:


# plotting/saving results
ft = 16
fig, ax = plt.subplots(1, 1)
ax.semilogx(border_08[0], border_08[1], label = '$\omega_c = 0.8$')
ax.semilogx(border_10[0], border_10[1], label = '$\omega_c = 1.0$')
ax.semilogx(border_12[0], border_12[1], label = '$\omega_c = 1.2$')
plt.xlabel('$g \sqrt{N_m}$', size=20)
plt.legend(loc = [0.6, 0.2], fontsize=ft-1, handlelength = 1)
plt.xticks(fontsize=ft, rotation=0)
plt.yticks(fontsize=ft, rotation=0)
ax.set_ylim(0.0, 6.0)

# plt.savefig('phot_pyth.png')

plt.show()

