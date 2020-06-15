
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, plot, show
import pickle

Gam_up_array = np.array([0.4, 1.0, 1.8]) * 1e-4
Gam_down = 1e-4  # excit decay


# In[18]:


# coupling_01 gives the results for matter-light coupling g \sqrt{N_m} = 0.1
# coupling_10 gives the results for matter-light coupling g \sqrt{N_m} = 1.0
# pumping_04, pumping_10, pumping_18 gives pumping 0.4 * 10^{-4} and so on
with open('spectrum_and_gain__coupling_01__pumping_18', 'rb') as handle:
    x1, y1, y11, x2, y2, bare_phot_with_A2, all_phot_comp_sort, i = pickle.load(handle)


# In[19]:


# plotting/saving results for the SPECTRUM
from matplotlib import rcParams
from matplotlib.collections import LineCollection

ft = 18


fig, ax1 = plt.subplots(figsize=(3.5, 2.7*1.2))

plt.yticks(fontsize = ft, rotation=0)
plt.xticks(fontsize = ft, rotation=0)

y = [0.0, 2.1]
plt.yticks(np.arange(min(y), max(y)+0.1, 0.5), ())

#plt.text(0.03, 1.8, '(' + let[i] + ')', fontsize = ft+2, color='k')
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



# In[24]:


## Density Matrix components of an unstable mode


# coupling_01 gives the results for matter-light coupling g \sqrt{N_m} = 0.1
# coupling_10 gives the results for matter-light coupling g \sqrt{N_m} = 1.0
# pumping_04, pumping_10, pumping_18 gives pumping 0.4 * 10^{-4} and so on
with open('DM_components__coupling_01__pumping_10', 'rb') as handle:
    phot_freq, rho_00, rho_10, rho_01 = pickle.load(handle)


# In[25]:


# plotting/saving the results
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.plot(phot_freq,  rho_00,  label = str(0) + "-" + str(0), color = 'k' )
ax.plot(phot_freq,  rho_10,  label = str(1) + "-" + str(0), color = 'r' )
ax.plot(phot_freq,  rho_01,  label = str(0) + "-" + str(1), color = 'b' )

#lett = 'd' + 'e' + 'f'

x = [0.0, 2.0]
plt.xticks(np.arange(min(x), max(x)+0.1, 0.5))

y = [0.0, 0.31]
plt.yticks(np.arange(min(y), max(y)+0.1, 0.1),())
plt.xlabel('$\omega_c$', size=ft+4)
#plt.legend(loc = [0.6, 0.2], fontsize = ft-2, handlelength = 1)
ax.set_ylabel('weight', fontsize = ft)
plt.xlim(0.0, 2.0)
plt.ylim(0, 0.3)

#plt.text(0.03, 0.26, '(' + lett[nn] + ')', fontsize = ft+2, color='k')

'''
plt.savefig('DM_comp_coupl' + str(g_light_mat).replace(".","") + 
            '__pump' + str(np.round(Gam_up_array[nn] / Gam_down, 1)).replace(".","") + 
            '.pdf', bbox_inches='tight', transparent=True)
'''

show(block=False)

