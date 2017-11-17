import tensorflow as tf
import numpy as np
import gym
import os
from gym.wrappers import Monitor


class Net(object):
	"""docstring for Net"""
	def __init__(self, layers, batch_size, learning_rate = 5e-3):
		self.layers = layers
		self.batch_size = batch_size
		self.learning_rate = learning_rate


	def build_model(self, size_of_state, num_of_actions, session):
		self.states = tf.placeholder(shape = [None, size_of_state], name = "states", dtype = tf.float32)
		self.actions = tf.placeholder(shape = [None], name = "actions_for_each_state", dtype = tf.int32)
		# self.rewards = tf.placeholder(shape = [None], name = "rewards_for_each_state", dtype = tf.float32)
		self.advantages = tf.placeholder(shape = [None], name = "Advantages_for_each_state", dtype = tf.float32)

		self.architecture = {}
		temp_input = self.states
		for index, num_units in enumerate(self.layers[:-1]):
			self.architecture[index] = tf.layers.dense(temp_input, num_units, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
			temp_input = self.architecture[index]
		# self.architecture[len(self.layers) - 1] is the output layer
		self.architecture[len(self.layers) - 1] = tf.layers.dense(temp_input, num_of_actions)
		logits = self.architecture[len(self.layers) - 1]
		self.probs = tf.nn.softmax(logits)
		self.log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.actions)
		self.loss = tf.reduce_mean(self.log_probs * self.advantages)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

		session.run(tf.global_variables_initializer())

	def train(self, session, actual_states, actual_actions, actual_advantages):
		feed_in = {self.states:actual_states, self.actions:actual_actions, self.advantages:actual_advantages}
		loss, _ = session.run([self.loss, self.optimizer], feed_in)
		return loss

def compute_advantage(rewards, gamma):
	adv = [0] * len(rewards)
	for index, reward in enumerate(rewards):
		if index == 0:
			adv[-1] = rewards[-1]
		adv[(-1 * index) - 1] += rewards[(-1 * index) - 1] + gamma*adv[-1 * index]

	adv = np.array(adv)
	mean = np.mean(adv)
	std = np.std(adv)

	adv = (adv - mean)/std

	for advan in adv:
		print(advan)
	return adv

def main(option):
	env = gym.make("CartPole-v0")
	size_of_state = env.observation_space.shape[0]
	# print(size_of_state)
	num_of_actions = env.action_space.n
	# print(num_of_actions)
	batch_size = 25
	gamma = 0.99
	session = tf.Session()

	my_net = Net([64, num_of_actions], batch_size)
	my_net.build_model(size_of_state, num_of_actions,session)

	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state(os.path.dirname('PG_checkpoints/'))
	if ckpt and ckpt.model_checkpoint_path:
		# print("hi")
		saver.restore(session, ckpt.model_checkpoint_path)


	if option == "play":
		state = env.reset()
		episode_reward = 0
		while True:
			env.render()
			feed_in = {my_net.states:np.array([state])}
			probs = session.run([my_net.probs], feed_in)[0][0]
			possible_actions = np.arange(num_of_actions)
			action_taken = np.random.choice(possible_actions, p = probs)
			state, reward, done, _ = env.step(action_taken)
			episode_reward += reward

			if done:
				break
	else:
		iterations = 1

		states, actions, rewards, advantages = [], [], [], []
		advantages = np.array(advantages)
		while iterations != 2000:
			state = env.reset()
			temp_rewards = []
			episode_reward = 0
			while True:
				feed_in = {my_net.states:np.array([state])}
				probs = session.run([my_net.probs], feed_in)[0][0]
				possible_actions = np.arange(num_of_actions)
				action_taken = np.random.choice(possible_actions, p = probs)
				actions.append(action_taken)
				states.append(state)
				state, reward, done, _ = env.step(action_taken)
				temp_rewards.append(reward)
				episode_reward += reward

				if done:
					rewards.append(temp_rewards)
					# advantages = advantages + compute_advantage(temp_rewards, gamma)
					advantages = np.concatenate(advantages, compute_advantage(temp_rewards, gamma))
					# print(episode_reward)
					saver.save(session, 'PG_checkpoints/PG')
					break

			iterations += 1
			if iterations % batch_size == 0:
				loss = my_net.train(session, states, actions, advantages)
				# print(" - ", loss)
				states, actions, rewards, advantages = [], [], [], []



if __name__ == '__main__':
	option = input("Play or Train?")
	main(option)