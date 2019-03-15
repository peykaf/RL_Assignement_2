#!/usr/bin/env python
# coding: utf-8

# # Taxi GYM
# 
# We start by importing some packages

# In[242]:


import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output



# Defining the agent

# In[243]:


class Agent():
    
    def __init__(self,algo,alpha,gamma,tau): #hyperparemeters
        self.algo = algo   #select which algorithm to use
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        
    def take_action(self,Q,state):
        probs = np.exp(Q[state]/self.tau)/sum(np.exp(Q[state]/self.tau)) #choose action based on a softmax policy
        choosen = np.random.choice(probs,p=probs)
        act = int(np.argwhere(probs == choosen)[0])
        return act,probs
    
    
    def update_qtable(self,Q,action,state,next_state,reward):
        
        if self.algo == 'Q-Learning':
            old_Q = Q[state,action]
            max_Q = np.max(Q[next_state])
            new_Q = old_Q + self.alpha*(reward + self.gamma * max_Q - old_Q)
            
        elif self.algo == 'SARSA':
            old_Q = Q[state,action]
            next_action = self.take_action(Q,next_state)[0]
            next_Q = Q[next_state, next_action]
            new_Q = old_Q + self.alpha*(reward + self.gamma * next_Q - old_Q)
            
            
        elif self.algo == 'EXP-SARSA':
            old_Q = Q[state,action]
            exp = self.take_action(Q,next_state)[1]
            next_Q = Q[next_state]
            new_Q = old_Q + self.alpha*(reward + self.gamma * np.dot(exp,next_Q) - old_Q)
        
        else: 
            raise Exception("Invalid algorithm")
        
        Q[state,action] = new_Q
        return Q
        
    
    def greedy_update(self,Q,action,state,next_state,reward):
        old_Q = Q[state,action]
        max_Q = np.max(Q[next_state])
        new_Q = old_Q + self.alpha*(reward + self.gamma * max_Q - old_Q)
        Q[state,action] = new_Q
        
        return Q


# In[244]:


def train(method,alpha = 0.1, gamma = 0.9, tau = 1 ):

    agent = Agent(method,alpha,gamma,tau)
    #initializing the environment and the action-value function table
    env = gym.make("Taxi-v2")
    Q = np.zeros([env.observation_space.n, env.action_space.n])



    for i in range(1, 991):
        state = env.reset()

        reward = 0
        done = False

        
        if i % 10 == 0:   ##act greedy
            while not done:
                action = np.argmax(Q[state])
                next_state, reward, done, info = env.step(action)
                Q = agent.greedy_update(Q,action,state,next_state,reward)
                state = next_state
        
        else:
            while not done:
                action = agent.take_action(Q,state)[0]
                next_state, reward, done, info = env.step(action)
                Q = agent.update_qtable(Q,action,state,next_state,reward)
                state = next_state



    
    def test(Q,agent):
        cum_reward = 0
            
        for j in range(10):   ##last 10 training episodes
            state = env.reset()
            reward = 0
            done = False

            while not done:

                action = agent.take_action(Q,state)[0]

                next_state, reward, done, info = env.step(action)

                Q = agent.update_qtable(Q,action,state,next_state,reward)

                state = next_state
                    
                cum_reward+=reward

        state = env.reset()
        reward, test_reward = 0, 0
        done = False
    

        while not done:
            action = np.argmax(Q[state])
            next_state, reward, done, info = env.step(action)
            Q = agent.update_qtable(Q,action,state,next_state,reward)
            state = next_state
            test_reward+=reward

        return cum_reward/10,test_reward

    return test(Q,agent)


# In[245]:


algos = ['Q-Learning','SARSA','EXP-SARSA']


taus = [0.5,1,1.5]
alphas = [0.1,0.35,0.60,0.85]
train_returns,test_returns = [],[]
for algo in algos:
    for tau in taus:
        for alpha in alphas:
            returns = []
            for _ in range(10):
                returns.append(train(method = algo,tau = tau, alpha = alpha))
            mean = np.mean(np.array(returns),axis = 0)
            print(f"method = {algo}, tau = {tau}, alpha = {alpha}. Avg return: {mean}")
            train_returns.append(mean[0])
            test_returns.append(mean[1])


# In[246]:


labels = []
for algo in algos:
    for tau in taus:
        for alpha in alphas:
            labels.append(f'{algo}, tau = {tau}')

labels = [labels[i] for i in range(0,35,4)]


# In[247]:


plot_train = np.reshape(np.array(train_returns),(9,4))
plot_test = np.reshape(np.array(test_returns),(9,4))
x = np.array(alphas)
plt.rcParams["figure.figsize"] = (20,15)
for k in range(3):
    plt.subplot(311)
    plt.plot(x,plot_train[k,:],label = labels[k])
    plt.ylabel('Average Reward')
    plt.xlabel('Learning Rate')
    plt.title('Avg. reward x Learning rate for training')    
    plt.legend()
    plt.subplot(312)
    plt.plot(x,plot_train[k+3,:],label = labels[k+3])
    plt.ylabel('Average Reward')
    plt.xlabel('Learning Rate')  
    plt.legend()
    plt.subplot(313)
    plt.plot(x,plot_train[k+6,:],label = labels[k+6])
    plt.ylabel('Average Reward')
    plt.xlabel('Learning Rate') 
    plt.legend()
plt.show()


# In[248]:


for k in range(3):
    plt.subplot(311)
    plt.plot(x,plot_test[k,:],label = labels[k])
    plt.ylabel('Average Reward')
    plt.xlabel('Learning Rate')
    plt.title('Avg. reward x Learning rate for testing')    
    plt.legend()
    plt.subplot(312)
    plt.plot(x,plot_test[k+3,:],label = labels[k+3])
    plt.ylabel('Average Reward')
    plt.xlabel('Learning Rate')  
    plt.legend()
    plt.subplot(313)
    plt.plot(x,plot_test[k+6,:],label = labels[k+6])
    plt.ylabel('Average Reward')
    plt.xlabel('Learning Rate') 
    plt.legend()
plt.show()


# The best hyperparameters for each method were:
# ### Q-Learning : $\tau = 1.5$, $\alpha = 0.60$
# ### SARSA : $\tau = 0.5$, $\alpha = 0.85$
# ### Expected-SARSA : $\tau = 1.0$, $\alpha = 0.60$

# In[234]:


def evaluate_best(method,alpha,tau,gamma=0.9):
    agent = Agent(method,alpha,gamma,tau)
    #initializing the environment and the action-value function table
    env = gym.make("Taxi-v2")
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    plot_rewards = []

    for i in range(1, 1001):
        state = env.reset()
        

        rewards = 0
        done = False

        while not done:
            action = agent.take_action(Q,state)[0]
            next_state, reward, done, info = env.step(action)
            Q = agent.update_qtable(Q,action,state,next_state,reward)
            state = next_state
            rewards+=reward
        

        plot_rewards.append(rewards)

    return plot_rewards


# In[255]:


best_returns = []
for _ in range(10):
    best_returns.append(evaluate_best(method = algos[0],tau = 1.5, alpha = 0.6))
mean = np.mean(np.array(best_returns),axis = 0)
std = np.std(np.array(best_returns),axis = 0)


# In[259]:


plt.plot(mean)
plt.show()


# In[257]:


plt.plot(std)


# In[206]:


pop = np.array([11,10,12])
print(np.exp(pop/1)/sum(np.exp(pop/1)))
print(np.exp(pop/1.5)/sum(np.exp(pop/1.5)))
print(np.exp(pop/0.5)/sum(np.exp(pop/0.5)))


# Results after 100 episodes:
# Average timesteps per episode: 12.71
# Average reward per episode: 0.08
