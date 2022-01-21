
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import os
import pickle

losses_fp  = pickle.load(open(os.path.join(os.getcwd(), 'Evaluation', 'LOSSES_4DVAR_SM_UPA_TD_AE_fp1it.pkl'), 'rb'))
losses_gbs = pickle.load(open(os.path.join(os.getcwd(), 'Evaluation', 'LOSSES_4DVAR_SM_UPA_TD_AE_gbs.pkl'), 'rb'))

losses_fp = np.mean(np.array(losses_fp).reshape(-1, 200), axis = 0)
losses_gbs = np.mean(np.array(losses_gbs).reshape(-1, 200), axis = 0)
# losses_fp = losses_fp[:200]
# losses_gbs = losses_gbs[:200]

fig, ax = plt.subplots(figsize = (6,4), dpi = 150)
ax.plot(losses_fp,  color = 'r', label = 'Fixed Point 1 iter')
ax.plot(losses_gbs, color = 'g', label = 'Gradient Solver')
ax.grid(axis = 'both', lw = 0.5)
ax.legend()
ax.set_yscale('log')
ax.set_xlabel('Epochs', fontsize = 14)
ax.set_ylabel('MSE', fontsize = 14)
# ax.set_title('4DVarNet FP1it: last L = {:.4f}\n4DVarNet Gradient: last L = {:.4f}'.format(losses_fp[-1].item(),
#                                                                    losses_gbs[-1].item()))
fig.savefig(os.path.join(os.getcwd(), 'losses.png'), format = 'png', dpi = 300, bbox_inches = 'tight')
plt.show(fig)