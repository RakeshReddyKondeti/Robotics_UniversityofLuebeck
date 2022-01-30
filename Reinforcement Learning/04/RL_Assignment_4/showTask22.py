import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

losses_tim = np.load('Results\\DDQN_losses1.npy')
losses_rak = np.load('Results\\DDQN_losses2.npy')
losses_chr = np.load('Results\\DDQN_losses3.npy')

reward_tim = np.load('Results\\DDQN_all_rewards1.npy')
reward_rak = np.load('Results\\DDQN_all_rewards2.npy')
reward_chr = np.load('Results\\DDQN_all_rewards3.npy')

x1 = np.arange(0, len(losses_tim), 1)
x2 = np.arange(0, len(reward_tim), 1)
x3 = np.arange(0, len(reward_rak), 1)
x4 = np.arange(0, len(reward_chr), 1)

plot1 = plt.figure(1)
plt.plot(x1, losses_tim, 'ro', markersize=2)
plt.plot(x1, losses_rak, 'go', markersize=2)
plt.plot(x1, losses_chr, 'bo', markersize=2)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('DDQN Loss')
plt.legend(['Tim', 'Rakesh', 'Christopher'])

arraylen = []
arraylen.append(len(reward_tim))
arraylen.append(len(reward_rak))
arraylen.append(len(reward_chr))
maxxval = np.max(arraylen)

plot2 = plt.figure(2)
plt.plot(x2, reward_tim, 'bo', markersize=2)
plt.plot([0, maxxval], [200, 200], 'k-')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('DDQN Reward Tim')
plt.legend(['Tim', '200 Reward'])

plot3 = plt.figure(3)
plt.plot(x3, reward_rak, 'bo', markersize=2)
plt.plot([0, maxxval], [200, 200], 'k-')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('DDQN Reward Rakesh')
plt.legend(['Rakesh', '200 Reward'])

plot4 = plt.figure(4)
plt.plot(x4, reward_chr, 'bo', markersize=2)
plt.plot([0, maxxval], [200, 200], 'k-')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('DDQN Reward Christopher')
plt.legend(['Christopher', '200 Reward'])

plt.show()