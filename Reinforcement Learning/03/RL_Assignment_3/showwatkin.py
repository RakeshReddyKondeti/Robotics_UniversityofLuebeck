import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

sarsa_1 = np.load('watkins_1.npy')
sarsa_2 = np.load('watkins_2.npy')
sarsa_3 = np.load('watkins_3.npy')
x = np.arange(0, 2000, 1)

# Sooth datasets an plot them
plt.plot(x, savgol_filter(sarsa_1, 39, 3), 'r-')
plt.plot(x, savgol_filter(sarsa_2, 39, 3), 'g-')
plt.plot(x, savgol_filter(sarsa_3, 39, 3), 'b-')
plt.axis([0, 2000, -1500, 0])
plt.xlabel('Episodes')
plt.ylabel('Return')
plt.title('Watkin')
plt.legend(['Tim', 'Rakesh', 'Christopher'])

plt.show()