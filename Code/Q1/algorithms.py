import gym
#import gym_gridworld
import gridworlds
import numpy as np
import matplotlib.pyplot as plt

COLORS = {0:[1.0,1.0,1.0], 1:[0.6,0.6,0.6], \
		2:[0.3,0.3,0.3], 3:[0.1,0.1,0.1], \
		4:[1.0,0.0,0.0], 5:[1.0,0.0,1.0], \
		7:[1.0,1.0,0.0]}

class Qlearning:

	def __init__(self, env, gamma = 0.9, alpha = 0.15, verbose = False):
		self.Q = np.zeros((env.observation_space.n, env.action_space.n))
		self.gamma = gamma
		self.alpha = alpha
		self.eps = 0.1
		self.env = env
		self.env.reset()
		self.env.verbose = verbose

	def get_action(self, state):
		if np.random.binomial(1, self.eps) == 1:
			return np.random.randint(0, self.env.action_space.n)

		mx = max(self.Q[state, :])
		idx = np.where(self.Q[state, :] == mx)[0]
		return np.random.choice(idx)
			

	def run(self, num_episodes):
		
		avg_steps = []
		avg_reward = []
		self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
		for i in range(0, int(num_episodes)):
			if (i % 50 == 0): print("Episode = {}".format(i))
			self.env.reset()
			curr_state = self.env.current_state()

			num_steps = 1.0
			tot_reward = 0

			while (self.env.done == False):
				curr_a = self.get_action(curr_state)
				nxt_state, r, d, _ = self.env.step(curr_a)
				tot_reward += r

				self.Q[curr_state, curr_a] = (1.0 - self.alpha) * self.Q[curr_state, curr_a] + self.alpha * (r + self.gamma * max(self.Q[nxt_state, :]))

				curr_state = nxt_state
				num_steps += 1.0
			
			avg_reward.append(tot_reward)
			avg_steps.append(num_steps)

		return np.array(avg_steps), np.array(avg_reward)


	def show_policy(self):
		# Should be called after self.run method
		policy_image = np.ones(self.env.obs_shape)
		gs0 = int(policy_image.shape[0] / self.env.grid_map_shape[0])
		gs1 = int(policy_image.shape[1] / self.env.grid_map_shape[1])
		for i in range(self.env.grid_map_shape[0]):
			for j in range(self.env.grid_map_shape[1]):
				for k in range(3):
					idx = self.env.start_grid_map[i, j]
					policy_image[i * gs0 : (i + 1) * gs0, j * gs1 : (j + 1) * gs1, k] = COLORS[idx][k]
					policy_image[i * gs0, j * gs1 : (j + 1) * gs1, k] = 0
					policy_image[(i + 1) * gs0, j * gs1 : (j + 1) * gs1, k] = 0
					policy_image[i * gs0 : (i + 1) * gs0, j * gs1, k] = 0
					policy_image[i * gs0 : (i + 1) * gs0, (j + 1) * gs1, k] = 0

		fig = plt.figure(3)
		plt.clf()
		plt.imshow(policy_image)

		rows = self.env.grid_map_shape[0]
		cols = self.env.grid_map_shape[1]
		for i in range(rows):
			for j in range(cols):
				action = np.argmax(self.Q[i * cols + j, :])
				if action == 0:
					plt.arrow((j + 1) * gs1 - gs1 / 2, (i + 1) * gs0 - gs0 / 4, 0, -gs0 / 2, head_width = 1.5, color = 'b', length_includes_head = True)
				elif action == 2:
					plt.arrow((j + 1) * gs1 - gs1 / 2, (i + 1) * gs0 - int(3.0 * gs0 / 4.0), 0, gs0 / 2, head_width = 1.5, color = 'b', length_includes_head = True)
				elif action == 1:
					plt.arrow(j * gs1 + gs1 / 4, (i + 1) * gs0 - gs0 / 2, gs1 / 2, 0, head_width = 1.5, color = 'b', length_includes_head = True)
				else:
					plt.arrow(j * gs1 + int(3.0 * gs1 / 4.0), (i + 1) * gs0 - gs0 / 2, -gs1 / 2, 0, head_width = 1.5, color = 'b', length_includes_head = True)

		plt.show()



class SARSA:

	def __init__(self, env, gamma = 0.9, alpha = 0.15, Lambda = 0, verbose = False):
		self.Q = np.zeros((env.observation_space.n, env.action_space.n)) 
		self.e = np.zeros((env.observation_space.n, env.action_space.n)) # eligibility trace
		self.gamma = gamma
		self.alpha = alpha
		self.Lambda = Lambda
		self.eps = 0.05
		self.env = env
		self.env.reset()
		self.env.verbose = verbose

	def get_action(self, state):
		if np.random.binomial(1, self.eps) == 1:
			return np.random.randint(0, self.env.action_space.n)
		mx = max(self.Q[state, :])
		idx = np.where(self.Q[state, :] == mx)[0]
		return np.random.choice(idx)

	def run(self, num_episodes):
		avg_steps = []
		avg_reward = []
		self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n)) 
		for i in range(num_episodes):
			print("Episode = {}".format(i))
			self.e = np.zeros((self.env.observation_space.n, self.env.action_space.n))
			self.env.reset()
			curr_state = self.env.current_state()
			curr_a = self.get_action(curr_state)
			num_steps = 1.0
			tot_reward = 0

			while (self.env.done == False):
				nxt_state, r, _, _ = self.env.step(curr_a)
				nxt_a = self.get_action(nxt_state)
				tot_reward += r

				if self.Lambda == 0:
					self.Q[curr_state, curr_a] = (1.0 - self.alpha) * self.Q[curr_state, curr_a] + self.alpha * (r + self.gamma * self.Q[nxt_state, nxt_a])
				else:
					TD_error = r + self.gamma * self.Q[nxt_state, nxt_a] - self.Q[curr_state, curr_a]
					self.e[curr_state, curr_a] += 1.0
					self.Q += self.alpha * TD_error * self.e
					self.e *= self.gamma * self.Lambda

				curr_state, curr_a = nxt_state, nxt_a
				num_steps += 1.0
			
			avg_reward.append(tot_reward)
			avg_steps.append(num_steps)

		return np.array(avg_steps), np.array(avg_reward)

	def show_policy(self):
		# Should be called after self.run method
		policy_image = np.ones(self.env.obs_shape)
		gs0 = int(policy_image.shape[0] / self.env.grid_map_shape[0])
		gs1 = int(policy_image.shape[1] / self.env.grid_map_shape[1])
		for i in range(self.env.grid_map_shape[0]):
			for j in range(self.env.grid_map_shape[1]):
				for k in range(3):
					idx = self.env.start_grid_map[i, j]
					policy_image[i * gs0 : (i + 1) * gs0, j * gs1 : (j + 1) * gs1, k] = COLORS[idx][k]
					policy_image[i * gs0, j * gs1 : (j + 1) * gs1, k] = 0
					policy_image[(i + 1) * gs0, j * gs1 : (j + 1) * gs1, k] = 0
					policy_image[i * gs0 : (i + 1) * gs0, j * gs1, k] = 0
					policy_image[i * gs0 : (i + 1) * gs0, (j + 1) * gs1, k] = 0

		fig = plt.figure(3)
		plt.clf()
		plt.imshow(policy_image)

		rows = self.env.grid_map_shape[0]
		cols = self.env.grid_map_shape[1]
		for i in range(rows):
			for j in range(cols):
				action = np.argmax(self.Q[i * cols + j, :])
				if action == 0:
					plt.arrow((j + 1) * gs1 - gs1 / 2, (i + 1) * gs0 - gs0 / 4, 0, -gs0 / 2, head_width = 1.5, color = 'b', length_includes_head = True)
				elif action == 2:
					plt.arrow((j + 1) * gs1 - gs1 / 2, (i + 1) * gs0 - int(3.0 * gs0 / 4.0), 0, gs0 / 2, head_width = 1.5, color = 'b', length_includes_head = True)
				elif action == 1:
					plt.arrow(j * gs1 + gs1 / 4, (i + 1) * gs0 - gs0 / 2, gs1 / 2, 0, head_width = 1.5, color = 'b', length_includes_head = True)
				else:
					plt.arrow(j * gs1 + int(3.0 * gs1 / 4.0), (i + 1) * gs0 - gs0 / 2, -gs1 / 2, 0, head_width = 1.5, color = 'b', length_includes_head = True)

		plt.show()


