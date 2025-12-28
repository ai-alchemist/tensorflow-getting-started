import gymnasium as gym
import numpy as np
import time

import matplotlib.pyplot as plt

########################################

####################
# Train an agent to navigate an environment.
# This example uses OpenAI "Gymnasium".
# For more details, see:
# https://github.com/Farama-Foundation/Gymnasium
####################

####################
# Building the Q-Table
####################

# load the FrozenLake environment.
# render_mode="ansi" renders actions as text.
# render_mode="human" renders actions as graphics.
env = gym.make('FrozenLake-v1', render_mode="ansi")

STATES = env.observation_space.n
ACTIONS = env.action_space.n

# create a matrix with all 0 values.
Q = np.zeros((STATES, ACTIONS))

# how many times to run the environment from the beginning.
EPISODES = 1500

# max number of steps allowed for each run of environment.
MAX_STEPS = 100

# learning rate.
LEARNING_RATE = 0.81
GAMMA = 0.96

# If you want to see training, set to True.
# This doesn't appear to have any effect in the modern
# version of this library.
# To *actually* disable rendering, set the render_mode to "ansi".
RENDER = False

####################
# Picking an Action
####################

# We can pick an action using one of two methods:
# 1.Randomly picking a valid action
# 2.Using the current Q-Table to find the best action.
# Here we will define a new value ϵ (epsilon) that will tell us
# the probability of selecting a random action.
# This value will start off very high and slowly decrease as the
# agent learns more about the enviornment.
epsilon = 0.9  # start with a 90% chance of picking a random action

rewards = []
for episode in range(EPISODES):

  state, _ = env.reset()
  for _ in range(MAX_STEPS):
    
    if RENDER:
      env.render()

    if np.random.uniform(0, 1) < epsilon:
      action = env.action_space.sample()  
    else:
      action = np.argmax(Q[state, :])

    #next_state, reward, done, _ = env.step(action)
    next_state, reward, done, truncated, info = env.step(action)

    Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

    state = next_state

    if done: 
      rewards.append(reward)
      epsilon -= 0.001
      break  # reached goal

print(Q)
print(f"Average reward: {sum(rewards)/len(rewards)}:")
# and now we can see our Q values!

# we can plot the training progress and see how the agent improved

def get_average(values):
  return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
  avg_rewards.append(get_average(rewards[i:i+100])) 

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()

