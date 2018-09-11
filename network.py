
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


    def __init__(self, state_size, action_size, learning_rate, name='PGNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate


        with tf.variable_scope(name):
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
            
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.actions)
            
            self.loss = tf.reduce_mean(self.neg_log_prob * self.rewards)
            
            self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
            
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
            self.gvs = self.optimizer.compute_gradients(self.loss, self.variables)

            self.placeholder_gradients = []
            for i, grad_var in enumerate(self.gvs):
                self.placeholder_gradients.append((tf.placeholder('float', shape=grad_var[1].get_shape()) ,grad_var[1]))
            
            self.train_opt = self.optimizer.apply_gradients(self.placeholder_gradients)

class ZNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='ZNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        with tf.variable_scope(name):
            with tf.name_scope("inputs"):
                # We create the placeholders
                # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
                # [None, 84, 84, 4]
                self.inputs_= tf.placeholder(tf.float32, [None, state_size], name="inputs_")
                self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
                self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards_")
            
            with tf.name_scope("fc1"):
                self.fc1 = tf.layers.dense(inputs = self.inputs_,
                                      units = 128,
                                      activation = tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="fc1")
           
          
            
            with tf.name_scope("logits"):
                self.logits = tf.layers.dense(inputs = self.fc1, 
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units = action_size, 
                                            activation=None)
            
            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(self.logits)
                

            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using 
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array. 
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.actions)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_) 
        
    
            with tf.name_scope("train"):
                self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)