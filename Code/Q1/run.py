from algorithms import Qlearning, SARSA
import matplotlib.pyplot as plt
import gridworlds
import numpy as np
import gym

env = gym.make('PuddleWorld-v0')


def get_derivables(learner, num_episodes, num_expts, show_policy = False, show_plots = True):
	''' num_episodes is number of episodes ''' 
	''' num_expts is number of experiments to average the performance over '''
	''' if show_policy is True, an image will be displayed showing optimal policy found by the agent after all episodes '''
	x_axis = np.arange(num_episodes)
	avg_steps = np.zeros(num_episodes)
	avg_reward = np.zeros(num_episodes)
	for i in range(num_expts):
		print ("Experiment number %d" % i)
		steps, rwd = learner.run(num_episodes)
		avg_steps += steps
		avg_reward += rwd
		if show_policy or i == num_expts - 1: 
			learner.show_policy()

	avg_steps /= float(num_expts)
	avg_reward /= float(num_expts)

	if (show_plots):
		plt.figure(1)
		plt.semilogy(x_axis, avg_steps)
		plt.ylabel("Average steps to reach the goal (log scale)")
		plt.xlabel("episodes")

		plt.figure(2)
		plt.plot(x_axis, avg_reward)
		plt.ylabel("Average reward per episode")
		plt.xlabel("episodes")

		plt.show()

	return avg_steps, avg_reward


get_derivables(Qlearning(env, verbose = False), 2000, 10, False)
#get_derivables(SARSA(env, Lambda = 0.7, verbose = False), 100, 1, False)

"""
avg_steps = []
avg_reward = []
lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for l in lambdas:
	print ("    Lambda = {}".format(l))
	steps, reward = get_derivables(SARSA(env, Lambda = l, verbose = False), 25, 10, show_plots = False)
	avg_steps.append(steps[24])
	avg_reward.append(reward[24])

plt.figure(3)
plt.plot(lambdas, avg_steps)
plt.title(r"Average steps after 25 episodes for different $\lambda$")
plt.ylabel("Average steps to reach the goal (averaged over 10 experiments)")
plt.xlabel(r"$\lambda$")

plt.figure(2)
plt.plot(lambdas, avg_reward)
plt.title(r"Average reward after 25 episodes for different $\lambda$")
plt.ylabel("Average reward per episode(averaged over 10 experiments)")
plt.xlabel(r"$\lambda$")

plt.show()
"""
