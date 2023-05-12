import matplotlib.pyplot as plt
import numpy as np


rad = np.linspace(0, 1, 10)
azm = np.linspace(-np.pi, np.pi, 360)
r, th = np.meshgrid(rad, azm)
z = (th ** 2.0) / 4.0


fig, axs = plt.subplots(1, 2, figsize=(10,5), subplot_kw={'projection':"polar"})

cb = axs[0].pcolormesh(th, r, z)

axs[0].plot(azm, r, color='k', ls='none') 
# plt.grid()
fig.colorbar(cb, ax=axs[0])
axs[0].set_yticks([],[])
axs[0].set_theta_offset(np.pi/2)

plt.show()