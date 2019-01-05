import numpy as np
from rlpa2 import chakra
import gym

class PolicyGradient:

	rng = np.random.RandomState(42)

	def __init__(self, env, gamma = 0.9, alpha_theta = 1e-2, alpha_omega = 0.1, verbose = False):
		env = env.unwrapped
		self.env = env
		self.env._reset()
		self.gamma = gamma
		self.alpha1 = alpha_theta
		self.alpha2 = alpha_omega
		self.action_dim = env.action_space.shape[0]
		self.obs_dim = env.observation_space.shape[0]
		self.verbose = verbose
		self.env.seed(42)
		print (env.current_state())

	def add_bias(self, state):
		return np.append(state, 1.0)

	def get_action(self, state):
		mean = self.theta.dot(self.add_bias(state))
		return np.random.multivariate_normal(mean = mean, cov = [[1.0, 0.0], [0.0, 1.0]])
		#return PolicyGradient.rng.normal(loc = mean, scale = 1.)

	def score_gradient(self, action, state):
		# score = log(policy) 
		b_state = self.add_bias(state)
		mean = self.theta.dot(b_state)
		action -= mean
		b_state = b_state.reshape(b_state.shape[0], 1)
		action = action.reshape(action.shape[0], 1)
		grad = action.dot(np.transpose(b_state))
		return grad

	def baseline_gradient(self, state):
		return np.zeros(state.shape[0] + 1)
		#return self.add_bias(state)  # Since we are going to subtract the gradient, we can skip returning the gradient with negative sign and instead add the returned gradient to the change

	def baseline(self, state):
		return self.omega.dot(self.add_bias(state))

	def run(self, max_iter, batch_size):
		self.grad = np.zeros((3))
		self.theta = np.random.normal(scale = 0.01, size = (self.action_dim, self.obs_dim + 1))  # Parameters of policy
		self.theta = np.zeros((self.action_dim, self.obs_dim + 1))  # Parameters of policy
		self.omega = np.random.normal(scale = 0.01, size = (self.obs_dim + 1))  # Parameters of baseline. PArameters of state value
		
		avg_return = []
		for i in range(max_iter):
			delta_theta = np.zeros((self.action_dim, self.obs_dim + 1))
			delta_omega = np.zeros(self.obs_dim + 1)
			avg_reward = 0.0
			print ("Iteration {}".format(i))
			self.env._reset()
			start_state = self.env.current_state()

			for j in range(batch_size):
				trajectory = []
				self.env.done = False;
				curr_state = start_state
				print ("Episode {} in batch {} with starting state {}".format(j, i, curr_state))
				return_avg = 0.0
				ret = 0.0
				g = 1.0

				while not self.env.done:
					if self.verbose:
						self.env._render()
					# An experience is a list - [current_state, current_action, reward_gained, next_state]
					curr_a = self.get_action(curr_state)
					nxt_state, rwd, done, _ = self.env._step(curr_a)
					#print (curr_state, self.theta.dot(self.add_bias(curr_state)))
					#print (curr_state, curr_a)
					ret = ret + g * rwd
					g *= self.gamma
					return_avg += ret
					trajectory.append([curr_state, curr_a, rwd, nxt_state]) # An experience is appended to the trajectory
					curr_state = nxt_state

				if (len(trajectory) == 0):
					continue
				return_avg /= float(len(trajectory))
				avg_reward += len(trajectory)
				
				Return = 0
				grad_theta = np.zeros((self.action_dim, self.obs_dim + 1))
				grad_omega = np.zeros(self.obs_dim + 1)
				for curr_exp in reversed(trajectory):
					curr_state, curr_a, rwd, _ = curr_exp
					Return = rwd + self.gamma * Return	
					advantage = Return - return_avg
					grad_theta += advantage * self.score_gradient(curr_a, curr_state) 
					#grad_omega += advantage * self.baseline_gradient(curr_state)

				grad_theta = grad_theta / (np.linalg.norm(grad_theta) + 1e-8)
				delta_theta += self.alpha1 * grad_theta
				#delta_omega += self.alpha2 * (grad_omega / (np.linalg.norm(grad_omega) + 1e-8))

			#delta_theta /= float(batch_size)   # Not needed. Just for the sake of completeness.
			
			avg_reward /= float(batch_size)
			avg_return.append(avg_reward)
			#delta_theta = delta_theta / (np.linalg.norm(delta_theta) + 1e-8)
			#delta_omega = delta_omega / (np.linalg.norm(delta_omega) + 1e-8)
			self.theta += delta_theta
			#self.omega += delta_omega

			print ("Iteration = {}; Avg reward = {};\ntheta = {}".format(i, avg_reward, self.theta))

		np.savetxt("./policies/policy_visham1.txt\n", self.theta, delimiter = ',')
		return np.array(avg_return)

	def simulate(self, theta, num_steps = None):
		if (num_steps == None):
			num_steps = 1e5
		while True:
			num_steps -= 1
			curr_state = self.env._reset()
			self.env._render()
			action = np.random.multivariate_normal(mean = theta.dot(self.add_bias(curr_state)), cov = [[1.0, 0.0], [0.0, 1.0]])
			nxt_state, rwd, done, _ = self.env.step(action)
			curr_state = nxt_state
			if (done):
				print ("Done. Did you see that?")

			if (num_steps == 0):
				break

