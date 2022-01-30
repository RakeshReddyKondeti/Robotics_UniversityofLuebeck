import numpy as np
import matplotlib.pyplot as plt

reward = np.load('/home/tim-henrik/Documents/Uni/Master/2. Semester/Reinforcement Learning/Assignments/04/Bonus_tasks_RL_Assignment_4/epi_returns_1.npy')

x1 = np.arange(0, len(reward), 1)

plot1 = plt.figure(1)
plt.plot(x1, reward, 'r-')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('DDQN Reward')
#plt.legend(['Tim', 'Rakesh', 'Christopher'])

plt.show()