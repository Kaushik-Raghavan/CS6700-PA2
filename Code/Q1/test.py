import gym
import gym_gridworld
import numpy as np

env = gym.make('WindyPuddleWorld-v0')
env.reset()
env.verbose = True

for _ in range(0, 300):
	env.render()
	action = env.action_space.sample()
	#print (env.state, action)
	st, rwd, d, _ = env.step(action)
	#print ("")
