import numpy as np
import tensorflow as tf
import os

from utils import openai_entropy, mse, LearningRateDecay

HIDDEN_SIZE = 512
FEATURE_SIZE = 4 * 4 * 128

def _conv_weight_variable(kernel_size, in_channels, out_channels, name='W_conv'):
		return tf.get_variable(name, shape=(kernel_size, kernel_size, in_channels, out_channels), \
			 initializer=tf.contrib.layers.xavier_initializer())

def _conv_bias_variable(size, name="W_b"):
	return tf.get_variable(name, shape=(size), initializer=tf.contrib.layers.xavier_initializer())

def _fc_weight_variable(shape, name='W_fc'):
	input_channels = shape[0]
	d = 1.0 / np.sqrt(input_channels)
	initial = tf.random_uniform(shape, minval=-d, maxval=d)
	return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

def _fc_bias_variable(shape, input_channels, name='b_fc'):
	d = 1.0 / np.sqrt(input_channels)
	initial = tf.random_uniform(shape, minval=-d, maxval=d)
	return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x) 

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

class Actor():
	def __init__(self, resolution, action_size, history_size=1, reuse = False):
		if history_size != 1:
			raise NotImplementedError("Stacking is not supported yet.")

		h, w, c = resolution
		self.action_size = action_size

		self.actions = tf.placeholder(tf.int32, [None, self.action_size])
		self.advantages = tf.placeholder(tf.float32, [None, ])

		with tf.variable_scope('Actor' if not reuse else "ShareLatent"):
			self.inputs = tf.placeholder(tf.float32, [None, h, w, c])


			self.W_conv1 = _conv_weight_variable(5, 3, 32, "W_conv1")
			self.b_conv1 = _conv_bias_variable(32, "b_conv1")

			self.W_conv2 = _conv_weight_variable(5, 32, 64, "W_conv2")
			self.b_conv2 = _conv_bias_variable(64, "b_conv2")

			self.W_conv3 = _conv_weight_variable(3, 64, 128, "W_conv3")
			self.b_conv3 = _conv_bias_variable(128, "b_conv3")

			self.W_conv4 = _conv_weight_variable(3, 128, 128, "W_conv4")
			self.b_conv4 = _conv_bias_variable(128, "b_conv4")

			self.conv1 = conv2d(self.inputs, self.W_conv1, self.b_conv1) # 128 x 128 x 32
			self.conv1 = maxpool2d(self.conv1, 4) # 32 x 32 x 32

			self.conv2 = conv2d(self.conv1, self.W_conv2, self.b_conv2) # 32 x 32 x 64
			self.conv2 = maxpool2d(self.conv2) # 16 x 16 x 64

			self.conv3 = conv2d(self.conv2, self.W_conv3, self.b_conv3) # 16 x 16 x 128
			self.conv3 = maxpool2d(self.conv3) # 8 x 8 x 128

			self.conv4 = conv2d(self.conv3, self.W_conv4, self.b_conv4) # 8 x 8 x 128
			self.conv4 = maxpool2d(self.conv4) # 4 x 4 x 128

			self.fc = tf.reshape(self.conv4, [-1, FEATURE_SIZE])

		with tf.variable_scope("Actions"):
			self.W_a = _fc_weight_variable([FEATURE_SIZE, self.action_size], name = "W_a")
			self.b_a = _fc_bias_variable([self.action_size], FEATURE_SIZE, name = "b_a")

		self.logits = tf.matmul(self.fc, self.W_a) + self.b_a

		self.pi = tf.nn.softmax(self.logits)
		self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.actions)
		self.policy_loss = tf.reduce_mean(self.neg_log_prob * self.advantages)

		# self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

class Critic():
	def __init__(self, resolution, history_size=1, reuse = False):
		if history_size != 1:
			raise NotImplementedError("Stacking is not supported yet.")

		h, w, c = resolution
		self.returns = tf.placeholder(tf.float32, [None, ])
		with tf.variable_scope('Critic' if not reuse else "ShareLatent" , reuse  = reuse):
			self.inputs = tf.placeholder(tf.float32, [None, h, w, c])


			self.W_conv1 = _conv_weight_variable(5, 3, 32, "W_conv1")
			self.b_conv1 = _conv_bias_variable(32, "b_conv1")

			self.W_conv2 = _conv_weight_variable(5, 32, 64, "W_conv2")
			self.b_conv2 = _conv_bias_variable(64, "b_conv2")

			self.W_conv3 = _conv_weight_variable(3, 64, 128, "W_conv3")
			self.b_conv3 = _conv_bias_variable(128, "b_conv3")

			self.W_conv4 = _conv_weight_variable(3, 128, 128, "W_conv4")
			self.b_conv4 = _conv_bias_variable(128, "b_conv4")

			self.conv1 = conv2d(self.inputs, self.W_conv1, self.b_conv1) # 128 x 128 x 32
			self.conv1 = maxpool2d(self.conv1, 4) # 32 x 32 x 32

			self.conv2 = conv2d(self.conv1, self.W_conv2, self.b_conv2) # 32 x 32 x 64
			self.conv2 = maxpool2d(self.conv2) # 16 x 16 x 64

			self.conv3 = conv2d(self.conv2, self.W_conv3, self.b_conv3) # 16 x 16 x 128
			self.conv3 = maxpool2d(self.conv3) # 8 x 8 x 128

			self.conv4 = conv2d(self.conv3, self.W_conv4, self.b_conv4) # 8 x 8 x 128
			self.conv4 = maxpool2d(self.conv4) # 4 x 4 x 128

			self.fc = tf.reshape(self.conv4, [-1, FEATURE_SIZE])

		with tf.variable_scope("Value", reuse = False):
			self.W_v = _fc_weight_variable([FEATURE_SIZE, 1], name = "W_v")
			self.b_v = _fc_bias_variable([1], FEATURE_SIZE, name = "b_v")

		self.value = tf.matmul(self.fc, self.W_v) + self.b_v
		self.value_loss = tf.reduce_mean(mse(tf.squeeze(self.value), self.returns))
   
		# self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
			
	def _fc_weight_variable(self, shape, name='W_fc'):
		input_channels = shape[0]
		d = 1.0 / np.sqrt(input_channels)
		initial = tf.random_uniform(shape, minval=-d, maxval=d)
		return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

	def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
		d = 1.0 / np.sqrt(input_channels)
		initial = tf.random_uniform(shape, minval=-d, maxval=d)
		return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

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
		self.name = name

		with tf.variable_scope(name):
			self.inputs= tf.placeholder(tf.float32, [None, self.state_size])
			self.actions = tf.placeholder(tf.int32, [None, self.action_size])
			self.rewards = tf.placeholder(tf.float32, [None, ])

			self.W_fc1 = _fc_weight_variable([self.state_size, 256], name='W_fc1')
			self.b_fc1 = _fc_bias_variable([256], self.state_size, name="b_fc1")
			self.fc1 = tf.nn.relu(tf.matmul(self.inputs, self.W_fc1) + self.b_fc1)

			self.W_fc2 = _fc_weight_variable([256, self.action_size], name='W_fc2')
			self.b_fc2 = _fc_bias_variable([self.action_size], 256, name="b_fc2")
			self.logits = tf.matmul(self.fc1, self.W_fc2) + self.b_fc2
		
			self.oracle = tf.nn.softmax(self.logits)

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
		if not os.path.isdir(os.path.join(save_dir, self.name)):
			os.makedirs(os.path.join(save_dir, self.name))
		save_path = os.path.join(save_dir, self.name, self.name)
		self.saver.save(sess, save_path)

	def restore_model(self, sess, save_dir):
		save_path = os.path.join(save_dir, self.name, self.name)
		self.saver.restore(sess, save_path)

class A2C():
	def __init__(self, 
				name, 
				resolution, 
				action_size, 
				history_size,
				entropy_coeff, 
				value_function_coeff, 
				max_gradient_norm, 
				joint_loss=False, 
				learning_rate=None, 
				decay=False, 
				reuse=False):

		self.name = name 
		self.max_gradient_norm  = max_gradient_norm
		self.entropy_coeff = entropy_coeff
		self.value_function_coeff = value_function_coeff
		self.resolution = resolution
		self.action_size = action_size
		self.reuse = reuse
		self.joint_loss = joint_loss

		# Add this placeholder for having this variable in tensorboard
		self.mean_reward = tf.placeholder(tf.float32)
		self.mean_redundant = tf.placeholder(tf.float32)
		self.total_mean_reward = tf.placeholder(tf.float32)
		self.average_mean_redundant = tf.placeholder(tf.float32)
		self.entropy_summary = tf.placeholder(tf.float32)
		
		with tf.variable_scope(name):
			self.actor = Actor(resolution=self.resolution, action_size=self.action_size, history_size=history_size, reuse=self.reuse)
			self.critic = Critic(resolution=self.resolution, history_size=history_size, reuse=self.reuse)

		self.learning_rate = tf.placeholder(tf.float32, [])
		self.fixed_lr = learning_rate
		self.decay = decay 

		with tf.variable_scope(name + '/actor_opt'):
			optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
			if self.max_gradient_norm is not None:
				gvs = optimizer.compute_gradients(self.actor.policy_loss)
				capped_gvs = [(tf.clip_by_value(grad, -self.max_gradient_norm, self.max_gradient_norm), var) for grad, var in gvs if grad is not None]
				self.train_opt_policy = optimizer.apply_gradients(capped_gvs)
			else:
				self.train_opt_policy = optimizer.minimize(self.actor.policy_loss)

		with tf.variable_scope(name + '/critic_opt'):
			optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
			if self.max_gradient_norm is not None:
				gvs = optimizer.compute_gradients(self.critic.value_loss)
				capped_gvs = [(tf.clip_by_value(grad, -self.max_gradient_norm, self.max_gradient_norm), var) for grad, var in gvs if grad is not None]
				self.train_opt_value = optimizer.apply_gradients(capped_gvs)
			else:
				self.train_opt_value = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.critic.value_loss)

		if self.joint_loss:

			self.entropy = tf.reduce_mean(openai_entropy(self.actor.logits))
			self.total_loss = self.actor.policy_loss + self.critic.value_loss * self.value_function_coeff - self.entropy * self.entropy_coeff

			with tf.variable_scope(name + '/joint_opt'):
				optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
				if self.max_gradient_norm is not None:
					gvs = optimizer.compute_gradients(self.total_loss)
					capped_gvs = [(tf.clip_by_value(grad, -self.max_gradient_norm, self.max_gradient_norm), var) for grad, var in gvs if grad is not None]
					self.train_opt_joint = optimizer.apply_gradients(capped_gvs)
				else:
					self.train_opt_joint = optimizer(self.learning_rate).minimize(self.total_loss)


	def set_lr_decay(self, lr_rate, nvalues):
		self.learning_rate_decayed = LearningRateDecay(v=lr_rate,
													   nvalues=nvalues,
													   lr_decay_method='linear')
		print("Learning rate decay-er has been set up!")

	def find_trainable_variables(self, key, printing = False):
		with tf.variable_scope(key):
			variables = tf.trainable_variables(key)
			if printing:
				print(len(variables), variables)
			return variables

	def save_model(self, sess, save_dir):
		if not os.path.isdir(save_dir):
			os.mkdir(save_dir)
		save_path = os.path.join(save_dir, self.name)
		self.saver.save(sess, save_path)

	def restore_model(self, sess, save_dir):
		save_path = os.path.join(save_dir, self.name)
		self.saver.restore(sess, save_path)
		
	def learn(self, sess, actor_states, critic_states, actions, returns, advantages):
		if self.decay:
			for i in range(len(actor_states)):
				current_learning_rate = self.learning_rate_decayed.value()
		else:
			current_learning_rate = self.fixed_lr

		feed_dict = {
						self.actor.inputs: actor_states, 
						self.critic.inputs: critic_states, 
						self.critic.returns: returns,
						self.actor.actions: actions, 
						self.actor.advantages: advantages,
						self.learning_rate: current_learning_rate,
					}

		if self.joint_loss:
			try:
				policy_loss, value_loss, policy_entropy, total_loss, _ = sess.run(
					[self.actor.policy_loss, self.critic.value_loss, self.entropy, self.total_loss, self.train_opt_joint],
					feed_dict = feed_dict
				)
			except ValueError:
				import sys
				print("Actor states: ", actor_states)
				print("Returns: ", returns)
				print("Actions: ", actions)
				print("Advantages: ", advantages)
				sys.exit()

			return policy_loss, value_loss, policy_entropy, total_loss
		else:
			policy_loss, value_loss, _, _ = sess.run(
				[self.actor.policy_loss, self.critic.value_loss, self.train_opt_policy, self.train_opt_value], 
				feed_dict = feed_dict)

			return policy_loss, value_loss, None, None

		
if __name__ == '__main__':
	a2c = A2C(100, 8, 0.05, 0.5, reuse = True)