import numpy as np
import gym
from rlpa2 import chakra, vishamC1
from algorithms import PolicyGradient
from matplotlib import pyplot as plt

env = gym.make('vishamC1-v0')
env._reset()
learner = PolicyGradient(env, gamma = 0.2, alpha_theta = 0.002, verbose = False)

max_iter = 150
batch_size = 50
avg_return = learner.run(max_iter, batch_size)

"""
param_file = open("./policies/policy6.txt").readlines()
theta = []
for lines in param_file:
	p1 = []
	for k in lines.split(','):
		p1.append(float(k))
	theta.append(p1)

theta = np.array(theta)
theta = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
learner.simulate(theta = theta)
"""

plt.figure(1)
plt.plot(np.arange(max_iter), avg_return)
plt.ylabel("Performance")
plt.xlabel("Iterations")
plt.show()
