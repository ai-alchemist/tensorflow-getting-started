import gymnasium as gym
import numpy as np
import time

########################################

####################
# Another basic example for how to use OpenAI "Gymnasium".
# This example uses OpenAI "Gymnasium".
# For more details, see:
# https://github.com/Farama-Foundation/Gymnasium
####################

####################
# Building the Q-Table
####################

# load the FrozenLake environment.
# render actions as text.
env = gym.make('FrozenLake-v1', render_mode="ansi")

STATES = env.observation_space.n
ACTIONS = env.action_space.n

# create a matrix with all 0 values.
Q = np.zeros((STATES, ACTIONS))

# how many times to run the environment from the beginning.
EPISODES = 2000

# max number of steps allowed for each run of environment.
MAX_STEPS = 100

# learning rate.
LEARNING_RATE = 0.81
GAMMA = 0.96

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

# code to pick action.
# check if a randomly selected value is less than epsilon.
if np.random.uniform(0, 1) < epsilon:
    # take random action. 
    action = env.action_space.sample()
else:
    # use Q table to pick best action based on current values.
    action = np.argmax(Q[state, :])

####################
# Updating Q Values
####################

# This code implements the Q-value formula.
Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]) - Q[state, action])
