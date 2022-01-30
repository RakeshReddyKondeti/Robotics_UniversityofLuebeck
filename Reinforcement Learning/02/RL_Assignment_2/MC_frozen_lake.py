# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:13:37 2020

@author: hongh
"""

from types import GetSetDescriptorType
from gym.logger import DEBUG
import numpy as np
import gym
import time

from numpy.core.fromnumeric import argmax




def epsilon_greedy(a,env,eps=0.05):
    # Input parma: 'a' : the greedy action for the currently-learned policy  
    # return the action index for the current state
    
    p = np.random.random()#First sample a value 'p' uniformly from the interval [0,1).
    
    if p < 1-eps:
        return a
    else:
        return np.random.randint(0, env.nA)

def interact_and_record(env,policy,EPSILON):
    # This function implements the sequential interaction of the agent to environement using decaying epsilon-greedy algorithm for a complete episode
    # It also records the necessary information e.g. state,action, immediate rewards in this episode.
    
    # Initilaize the environment, returning s = S_0
    s = env.reset()     
    state_action_reward = []
    
    # start interaction
    while True:
        a = epsilon_greedy(policy[s],env,eps=EPSILON)
        # Agent interacts with the environment by taking action a in state s,\  env.step()
        # receiving successor state s_, immediate reward r, and a boolean variable 'done' telling if the episode terminates.
        # You could print out each variable to check in details.
        s_,r,done,_ = env.step(a)
        # store the <s,a,immediate reward> in list for each step
        state_action_reward.append((s,a,r))
        if done:            
            break        
        s=s_ 
    
    
    G=0
    state_action_return = []
    state_action_trajectory = []

    for step in state_action_reward[::-1]:
        G = GAMMA * G + step[2]

        state_action_return.append((step[0], step[1], G))
        state_action_trajectory.append((step[0], step[1]))
    
    #state_action_trajectory = state_action_trajectory.reverse()

    # TO DO : Compute the return G for each state-action pair visited
    # Hint : You need to compute the return in reversed order, first from the last state-action pair, finally to (s_0,a_0)
    # Return : (1) state_action_return = [(S_(T-1), a_(T-1), G_(T-1)), (S_(T-2), a_(T-2), G_(T-2)) ,... (S_0,a_0.G_0)]
    # (2) state_action_trajectory = [(s_0, a_0), (s_1,a_1), ... (S_(T-1)), a_(T-1))] , note:  the order is different
    # Note: even if (s_n,a_n) is encountered multiple times in an episode, here we still store them in the list, checking if it is the first appearance is done in def monte_carlo()
    return state_action_return, state_action_trajectory

    
def monte_carlo(env, N_EPISODES):
    if DEBUG: print("Initializing Monte Carlo...")

    # Initialize the random policy , useful function: np.random.choice()  env.nA, env.nS
    policy = np.random.choice(env.nA, env.nS) # an 1-D array of the length = env.nS
    
    a2w = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = np.array([a2w[x] for x in policy])
    print("Initial random policy:")
    print(np.array(policy_arrows).reshape([-1, 4]))

    # To do : Intialize the Q table and number of visit per state-action pair to 0 using np.zeros()
    Q = np.zeros((env.nS, env.nA)) 
    visit = np.zeros((env.nS, env.nA))


    if DEBUG: print("Starting Monte Carlo learning...")    
    # MC approaches start learning
    for i in range(N_EPISODES):
        if DEBUG and i % 10000 == 0: print("MC Episode " + str(i))

        # epsilon-greedy exploration strategy 
        epsilon = 0.05
        # Interact with env and record the necessary info for one episode.
        state_action_return, state_action_trajectory = interact_and_record(env,policy,epsilon)

        visit_this_episode = np.zeros((env.nS, env.nA))
      
        count_episode_length = 0 #
        if DEBUG and i % 10000 == 0: print("Calculating new Policy...", end="")

        for s,a,G in state_action_return:
            count_episode_length += 1
            
            # Check whether s,a is the first appearnace and perform the update of Q values
            if visit_this_episode[s][a] == 0:
                visit_this_episode[s][a] = 1
                visit[s][a] += 1

                #Update Q-Values
                Q[s][a] = Q[s][a] + (G - Q[s][a]) / visit[s][a]

            # Update policy for the current state
            policy[s] = np.argmax(Q[s])
        
        if DEBUG and i % 10000 == 0: print(" Done!")
        if DEBUG and i % 10000 == 0: print("Current Policy:")
        if DEBUG and i % 10000 == 0: policy_arrows = np.array([a2w[x] for x in policy])
        if DEBUG and i % 10000 == 0: print(np.array(policy_arrows).reshape([-1, 4]))
    # Return   the finally learned policy , and the number of visits per state-action pair

    if DEBUG: print("Monte Carlo learning done!")
    return policy, visit

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
    DEBUG = False
    if DEBUG: print("Debug ON")
    
    env = gym.make('FrozenLake-v0')
    random_seed = 13333 # Don't change
    N_EPISODES = 150000 # Don't change
    if random_seed:
        env.seed(random_seed)
        np.random.seed(random_seed)    
    GAMMA = 1.0
    start = time.time()
    
    policy,visit = monte_carlo(env,N_EPISODES=N_EPISODES)#EPSILON=1.0,N_EPISODES=N_EPISODES)
    print('TIME TAKEN {} seconds'.format(time.time()-start))
    a2w = {0:'<', 1:'v', 2:'>', 3:'^'}
    # Convert the policy action into arrows
    policy_arrows = np.array([a2w[x] for x in policy])
    # Display the learned policy
    print(np.array(policy_arrows).reshape([-1, 4]))

    get_score(env, policy)