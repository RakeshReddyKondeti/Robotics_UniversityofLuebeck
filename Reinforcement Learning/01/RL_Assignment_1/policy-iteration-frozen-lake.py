# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:32:22 2020

@author: hongh
"""
import numpy as np
import matplotlib.pyplot as plt
import gym
import time

start_time = time.time()


def policy_iter(env, gamma, theta):
    """To Do : Implement Policy Iteration Algorithm
    gamma (float) - discount factor
    theta (float) - termination condition
    env - environment with following required memebers:
    
    Useful variables/functions:
        
            env.nb_states - number of states
            env.nb_action - number of actions
            env.model     - prob-transitions and rewards for all states and actions, you can play around that
        
        
        return the value function V and policy pi, 
        pi should be a determinstic policy and an illustration of randomly initialized policy is below
    """
    # initialize the random policy
    pi = np.random.randint(low=0, high=env.action_space.n, size=env.nb_states)
    # Initialize the value function
    V = np.zeros(env.nb_states)
    counter = 0
    
    while True:
    
        # 2. Policy Evaluation
        while True:
            delta = 0
            for s in range(env.nb_states):
                v = V[s]
                V[s] = sum_sr(env, V=V, s=s, a=pi[s], gamma=gamma)
                delta = max(delta, abs(v - V[s]))
            if (counter > 1000) and (delta < theta): break

            counter = counter+1
            print(counter)

        # 3. Policy Improvement
        policy_stable = True
        for s in range(env.nb_states):
            old_action = pi[s]
            pi[s] = np.argmax([sum_sr(env, V=V, s=s, a=a, gamma=gamma)  # list comprehension
                               for a in range(env.nb_actions)])
            if old_action != pi[s]: policy_stable = False
        if policy_stable: break
    
    return V, pi

def sum_sr(env, V, s, a, gamma):
    """Calc state-action value for state 's' and action 'a'"""
    tmp = 0  # state value for state s
    for p, s_, r, _ in env.model[s][a]:     # see note #1 !
        # p  - transition probability from (s,a) to (s')
        # s_ - next state (s')
        # r  - reward on transition from (s,a) to (s')
        tmp += p * (r + gamma * V[s_])
    return tmp

def get_score(env, policy, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps=0
        while True:
        
            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps+=1
            if done and reward == 1:
                # print('You have got the fucking Frisbee after {} steps'.format(steps))
                steps_list.append(steps)
                break
            elif done and reward == 0:
                # print("You fell in a hole!")
                misses += 1
                break
    print('----------------------------------------------')
    print('You took an average of {:.0f} steps to get the frisbee'.format(np.mean(steps_list)))
    print('And you fell in the hole {:.2f} % of the times'.format((misses/episodes) * 100))
    print('----------------------------------------------')

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env.reset()
    env.render()
    
    
    if not hasattr(env, 'nb_states'):  
        env.nb_states = env.env.nS
    if not hasattr(env, 'nb_actions'): 
        env.nb_actions = env.env.nA
    if not hasattr(env, 'model'):      
        env.model = env.env.P

    # Check #state, #actions and transition model
    # env.model[state][action]
    print(env.nb_states, env.nb_actions, env.model[14][2])


    # display the result    
    V, pi = policy_iter(env, gamma=1.0, theta=1e-5)
    print(V.reshape([4, -1]))    
    
    a2w = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = np.array([a2w[x] for x in pi])
    print(np.array(policy_arrows).reshape([-1, 4]))

    print("--- %s seconds ---" %(time.time() - start_time))

    get_score(env, pi, episodes=100)
    
#    print(env.model[14][2])