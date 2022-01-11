import torch
import numpy as np
import matplotlib.pyplot as plt

data = torch.load('test_result')

# plot energy correlation
fig, ax = plt.subplots(1, 2, subplot_kw={"aspect": "equal"})
dft_energy = np.copy(data['DFT_E'])
nnp_energy = np.copy(data['NN_E'])
tot_num = np.copy(data['N'])
ax[0].scatter(dft_energy / tot_num, nnp_energy / tot_num)
lim = ax[0].get_xlim()
ax[0].plot(lim, lim, color='k', alpha=0.75, zorder=0)
ax[0].set_xlabel("$E^{\mathrm{DFT}} (\mathrm{eV}$/atom)")
ax[0].set_ylabel("$E^{\mathrm{NNP}} (\mathrm{eV}$/atom)")

# plot force correlation
dft_force = np.concatenate(data['DFT_F'])
nnp_force = np.concatenate(data['NN_F'])
ax[1].scatter(dft_force.flatten(), nnp_force.flatten())
lim = ax[1].get_xlim()
ax[1].plot(lim, lim, color='k', alpha=0.75, zorder=0)
ax[1].set_xlabel("$F^{\mathrm{DFT}} (\mathrm{eV/\AA}$)")
ax[1].set_ylabel("$F^{\mathrm{NNP}} (\mathrm{eV/\AA}$)")

fig.tight_layout()
fig.savefig("result.png")
