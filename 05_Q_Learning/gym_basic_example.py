import gymnasium as gym
import numpy as np
import time

########################################

####################
# Basic example for how to use OpenAI "Gymnasium".
# For more details, see:
# https://github.com/Farama-Foundation/Gymnasium
####################

# load the FrozenLake environment.
# render actions as text.
env = gym.make('FrozenLake-v1', render_mode="ansi")

# print number of states
print(env.observation_space.n)

# print number of actions.
print(env.action_space.n)

# reset enviornment to default state.
env.reset()

# get a random action.
action = env.action_space.sample()

# take action, and get information about the action
new_state, reward, done, truncated, info = env.step(action)

# render the GUI for the environment.
# this prints a grid, which represents the "game board".
# "S" is the starting point.
# "F" are frozen (safe) locations.
# "H" are holes (forbidden) locations.
# "G" is the goal.
print(env.render())

