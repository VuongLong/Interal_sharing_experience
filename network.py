import tensorflow as tf
import numpy as np           # Handle matrices
import random                # Handling random number generation
import time                  # Handling time calculation
import math


class PGNetwork:

	def _fc_weight_variable(self, shape, name='W_fc'):
		input_channels = shape[0]
		d = 1.0 / np.sqrt(input_channels)
		initial = tf.random_uniform(shape, minval=-d, maxval=d)
		return tf.Variable(initial, name=name)

	def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
		d = 1.0 / np.sqrt(input_channels)
		initial = tf.random_uniform(shape, minval=-d, maxval=d)
		return tf.Variable(initial, name=name)

	def __init__(self, state_size, task_size, action_size, learning_rate, name='PGNetwork'):
		self.state_size = state_size
		self.task_size = task_size
		self.action_size = action_size
		self.learning_rate = learning_rate
		self.name = name

		with tf.variable_scope(self.name):
			self.inputs= tf.placeholder(tf.float32, [None, self.state_size])
			self.actions = tf.placeholder(tf.int32, [None, self.action_size])
			self.rewards = tf.placeholder(tf.float32, [None, ])
			
			# Add this placeholder for having this variable in tensorboard
			self.mean_reward = tf.placeholder(tf.float32)
			
			self.W_fc1 = self._fc_weight_variable([self.state_size, 256])
			self.b_fc1 = self._fc_bias_variable([256], self.state_size)
			self.fc1 = tf.nn.relu(tf.matmul(self.inputs, self.W_fc1) + self.b_fc1)

			self.W_fc2 = self._fc_weight_variable([256, self.action_size])
			self.b_fc2 = self._fc_bias_variable([self.action_size], 256)
			self.logits = tf.matmul(self.fc1, self.W_fc2) + self.b_fc2
			
			self.pi = tf.nn.softmax(self.logits)
			
			self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.actions)

			self.loss = tf.reduce_mean(self.neg_log_prob * self.rewards)

			self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
		self.saver = tf.train.Saver(self.get_vars())

	def get_vars(self):
		return [
			self.W_fc1, self.b_fc1,
			self.W_fc2, self.b_fc2
		]
	
	def save_model(self, sess, save_dir):
		save_path = "{}/{}".format(save_dir, self.name)
		self.saver.save(sess, save_path)
	
	def restore_model(self, sess, save_dir):
		save_path = "{}/{}".format(save_dir, self.name)
		self.saver.restore(sess, save_path)
		

class VNetwork:

	def _fc_weight_variable(self, shape, name='W_fc'):
		input_channels = shape[0]
		d = 1.0 / np.sqrt(input_channels)
		initial = tf.random_uniform(shape, minval=-d, maxval=d)
		return tf.Variable(initial, name=name)

	def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
		d = 1.0 / np.sqrt(input_channels)
		initial = tf.random_uniform(shape, minval=-d, maxval=d)
		return tf.Variable(initial, name=name)

	def __init__(self, state_size, task_size, action_size, learning_rate, name='VNetwork'):
		self.state_size = state_size
		self.task_size = task_size
		self.action_size = action_size
		self.learning_rate = learning_rate
		self.name = name

		with tf.variable_scope(self.name):
			self.inputs= tf.placeholder(tf.float32, [None, self.state_size])
			self.rewards = tf.placeholder(tf.float32, [None, ])
		
			self.W_fc1 = self._fc_weight_variable([self.state_size, 256])
			self.b_fc1 = self._fc_bias_variable([256], self.state_size)
			self.fc1 = tf.nn.relu(tf.matmul(self.inputs, self.W_fc1) + self.b_fc1)

			self.W_fc2 = self._fc_weight_variable([256, 1])
			self.b_fc2 = self._fc_bias_variable([1], 256)
			self.value = tf.matmul(self.fc1, self.W_fc2) + self.b_fc2
			
			self.loss = 0.5 * tf.reduce_mean(tf.square(self.rewards - tf.reshape(self.value,[-1])))
			self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
		self.saver = tf.train.Saver(self.get_vars())	

	def get_vars(self):
		return [
			self.W_fc1, self.b_fc1,
			self.W_fc2, self.b_fc2
		]

	def save_model(self, sess, save_dir):
		save_path = "{}/{}".format(save_dir, self.name)
		self.saver.save(sess, save_path)
	
	def restore_model(self, sess, save_dir):
		save_path = "{}/{}".format(save_dir, self.name)
		self.saver.restore(sess, save_path)


class ZNetwork:

	def _fc_weight_variable(self, shape, name='W_fc'):
		input_channels = shape[0]
		d = 1.0 / np.sqrt(input_channels)
		initial = tf.random_uniform(shape, minval=-d, maxval=d)
		return tf.Variable(initial, name=name)

	def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
		d = 1.0 / np.sqrt(input_channels)
		initial = tf.random_uniform(shape, minval=-d, maxval=d)
		return tf.Variable(initial, name=name)

	def __init__(self, state_size, action_size, learning_rate, name='ZNetwork'):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate

		with tf.variable_scope(name):
			self.inputs= tf.placeholder(tf.float32, [None, self.state_size])
			self.actions = tf.placeholder(tf.int32, [None, self.action_size])
			self.rewards = tf.placeholder(tf.float32, [None, ])

			self.W_fc1 = self._fc_weight_variable([self.state_size, 256])
			self.b_fc1 = self._fc_bias_variable([256], self.state_size)
			self.fc1 = tf.nn.relu(tf.matmul(self.inputs, self.W_fc1) + self.b_fc1)

			self.W_fc2 = self._fc_weight_variable([256, self.action_size])
			self.b_fc2 = self._fc_bias_variable([self.action_size], 256)
			self.logits = tf.matmul(self.fc1, self.W_fc2) + self.b_fc2
		
			self.oracle = tf.nn.softmax(self.logits)

			self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.actions)
			
			self.loss = tf.reduce_mean(self.neg_log_prob * self.rewards)
			
			self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
