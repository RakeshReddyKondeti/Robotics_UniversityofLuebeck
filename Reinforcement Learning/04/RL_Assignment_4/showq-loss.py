import numpy as np
import matplotlib.pyplot as plt

estQrun = np.load('Results\\DQN_estQ_running_network1.npy')
estQtar = np.load('Results\\DQN_estQ_target_network1.npy')

q_loss = np.load('Results\\losses1.npy')

x1 = np.arange(0, len(estQrun), 1)
x2 = np.arange(0, len(estQtar), 1)
x3 = np.arange(0, len(q_loss), 1)

plot1 = plt.figure(1)
plt.plot(x1, estQrun, 'r-')
plt.plot(x2, estQtar, 'b-')
plt.xlabel('Steps')
plt.ylabel('Q-value')
plt.title('Estimated Q value of target and running system')
plt.legend(['Running', 'Target'])

plot2 = plt.figure(2)
plt.plot(x3, q_loss)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss of the system')

plt.show()