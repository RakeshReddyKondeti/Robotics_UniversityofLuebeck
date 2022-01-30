# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:13:02 2020

@author: hongh
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
import time

start_time = time.time()

env = gym.make('FrozenLake-v0')


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

def value_iter(env, gamma, theta):
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
    # Initialize the value function
    V = np.zeros(env.nb_states)
    counter = 0
    
    stateValue = [0 for i in range(env.nS)]
    newStateValue = stateValue.copy()
    while True:
        for state in range(env.nS):
            action_values = []      
            for action in range(env.nA):
                state_value = 0
                for i in range(len(env.P[state][action])):
                    prob, next_state, reward, done = env.P[state][action][i]
                    state_action_value = prob * (reward + gamma*stateValue[next_state])
                    state_value += state_action_value
                    action_values.append(state_value)      #the value of each action
                    best_action = np.argmax(np.asarray(action_values))   # choose the action which gives the maximum value
                    newStateValue[state] = action_values[best_action]  #update the value of the state
        if counter > 1000: 
            if sum(stateValue) - sum(newStateValue) < theta:   # if there is negligible difference break the loop
                break
        else:
            stateValue = newStateValue.copy()
        counter = counter+1
        print(counter)

    V = np.array(stateValue)

    policy = [0 for i in range(env.nS)]
    for state in range(env.nS):
        action_values = []
        for action in range(env.nA):
            action_value = 0
            for i in range(len(env.P[state][action])):
                prob, next_state, r, _ = env.P[state][action][i]
                action_value += prob * (r + gamma * stateValue[next_state])
            action_values.append(action_value)
        best_action = np.argmax(np.asarray(action_values))
        policy[state] = best_action

    pi = policy

    return V, pi



if __name__ == '__main__':
    env.reset()
    env.render()

    # Check #state, #actions and transition model
    # env.model[state][action]
    #print(env.nb_states, env.nb_actions, env.model[14][2])
    
    if not hasattr(env, 'nb_states'):  
        env.nb_states = env.env.nS
    if not hasattr(env, 'nb_actions'): 
        env.nb_actions = env.env.nA
    if not hasattr(env, 'model'):      
        env.model = env.env.P
        
  
    V, pi = value_iter(env, gamma=1.0, theta=1e-4)
    print(V.reshape([4, -1]))
    
    
    a2w = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = np.array([a2w[x] for x in pi])
    print(np.array(policy_arrows).reshape([-1, 4]))

    print("--- %s seconds ---" %(time.time() - start_time))

    get_score(env, pi, episodes=100)