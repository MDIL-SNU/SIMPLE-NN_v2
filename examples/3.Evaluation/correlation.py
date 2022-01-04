import torch
import numpy as np
import matplotlib.pyplot as plt

data = torch.load('test_result')

# plot energy correlation
fig1, ax1 = plt.subplots(1)
ax1.scatter(data['DFT_E'], data['NN_E'])

lim1 = ax1.get_xlim()
ax1.plot(lim1, lim1, color='k', alpha=0.75, zorder=0)

# plot force correlation
fig2, ax2 = plt.subplots(1)
tmp = np.concatenate(data['DFT_F'])
dft_f = np.concatenate(tmp)
tmp = np.concatenate(data['NN_F'])
nn_f = np.concatenate(tmp)
ax2.scatter(dft_f, nn_f)

lim2 = ax2.get_xlim()
ax2.plot(lim2, lim2, color='k', alpha=0.75, zorder=0)
plt.show()
