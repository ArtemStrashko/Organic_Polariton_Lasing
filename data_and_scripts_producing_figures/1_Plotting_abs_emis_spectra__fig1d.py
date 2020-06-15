
# coding: utf-8

# In[4]:


import pylab as plt
import pickle


# In[10]:


with open('abs_sp', 'rb') as handle:
    abs_spectrum = pickle.load(handle)
    
with open('emis_sp', 'rb') as handle:
    emis_spectrum = pickle.load(handle)


# In[13]:


# plot the spectra
ft = 14
plt.figure(figsize=(3.5, 2.5))
plt.plot(abs_spectrum[0], abs_spectrum[1], label = 'abs')
plt.plot(emis_spectrum[0], emis_spectrum[1], label = 'emis')
plt.xlim(0.0, 2.0)
plt.ylim(0, 1)
plt.legend(loc = [0.6, 0.62], fontsize=14, handlelength = 1)
plt.xlabel('$\omega_c$', size=ft)
plt.xticks(fontsize = ft, rotation=0)
plt.yticks(fontsize = ft, rotation=0)

x = [0.0, 2.0]
y = [0.0, 1.0]
plt.xticks(np.arange(min(x), max(x)+0.1, 0.4))
plt.yticks(np.arange(min(y), max(y)+0.1, 0.2))

'''
plt.savefig('abs_emis__S' + str(S).replace(".","") + 
                '__N_' + str(N) + '.svg', bbox_inches='tight', transparent=True)
'''                

plt.show(block=False)
               

plt.show(block=False)

