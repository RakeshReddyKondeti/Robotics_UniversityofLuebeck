import numpy as np
import matplotlib.pyplot as plt

sarsa_Q05 = np.load('SARSA_0.5_5000_Q.npy')
sarsa_Q075 = np.load('SARSA_0.75_5000_Q.npy')
sarsa_Q095 = np.load('SARSA_0.95_5000_Q.npy')
sarsa_Q099 = np.load('SARSA_0.99_5000_Q.npy')

sarsa_Q05 = np.max(sarsa_Q05, axis=2)
sarsa_Q075 = np.max(sarsa_Q075, axis=2)
sarsa_Q095 = np.max(sarsa_Q095, axis=2)
sarsa_Q099 = np.max(sarsa_Q099, axis=2)

sarsa_Q05 = sarsa_Q05/sarsa_Q05.max()
sarsa_Q075 = sarsa_Q075/sarsa_Q075.max()
sarsa_Q095 = sarsa_Q095/sarsa_Q095.max()
sarsa_Q099 = sarsa_Q099/sarsa_Q099.max()

fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4)
ax1.imshow(sarsa_Q05, vmin=0, vmax=1)
ax2.imshow(sarsa_Q075, vmin=0, vmax=1)
ax3.imshow(sarsa_Q095, vmin=0, vmax=1)
image = ax4.imshow(sarsa_Q099, vmin=0, vmax=1)

fig.suptitle('Q matrices of different lambdas')
ax1.set_title('lambda=0.5')
ax2.set_title('lambda=0.75')
ax3.set_title('lambda=0.95')
ax4.set_title('lambda=0.99')

fig.colorbar(image)
plt.show()