import tensorflow as tf      			# Deep Learning library
import numpy as np           			# Handle matrices
import random                			# Handling random number generation
import time                  			# Handling time calculation
import math
import copy
import threading
import os

from rollout import Rollout
from env.map import ENV_MAP
from plot_figure import PlotFigure
from collections import deque			# Ordered collection with ends
from env.terrain import Terrain

class MultitaskPolicy(object):

	def __init__(
			self,
			map_index,
			policy,
			oracle,
			writer,
			write_op,
			action_size,
			num_task,
			num_epochs,			
			gamma,
			plot_model,
			save_model,
			save_name,
			num_episode,
			share_exp,
			combine_gradent,
			share_exp_weight
			):

		self.map_index = map_index
		self.PGNetwork = policy
		self.ZNetwork = oracle

		self.writer = writer
		self.write_op = write_op
		self.action_size = action_size
		self.num_task = num_task
		self.num_epochs = num_epochs
		self.gamma = gamma
		self.save_name = save_name
		self.plot_model = plot_model
		self.save_model = save_model

		self.num_episode =  num_episode
		self.combine_gradent = combine_gradent
		self.share_exp = share_exp
		self.share_exp_weight = share_exp_weight

		self.env = Terrain(self.map_index)

		assert self.num_task <= self.env.num_task

		self.plot_figure = PlotFigure(self.save_name, self.env, self.num_task)

		self.rollout = Rollout(number_episode = self.num_episode, num_task = self.num_task, map_index = self.map_index)


	def _discount_rewards(self, episode_rewards):
		discounted_episode_rewards = np.zeros_like(episode_rewards)
		cumulative = 0.0
		for i in reversed(range(len(episode_rewards))):
			cumulative = cumulative * self.gamma + episode_rewards[i]
			discounted_episode_rewards[i] = cumulative

		return discounted_episode_rewards.tolist()



	def _prepare_current_policy(self, sess, epoch):
		current_policy = {}
		current_Z = {}

		# env.bounds_x = [1, 17]
		# env.bounds_y = [1, 5]
		# env.cv_state_onehot : (17x5, 17x5)

		for task in range(self.num_task):
			for x in range(1, self.env.bounds_x[1]+1):
				for y in range(1, self.env.bounds_y[1]+1):
					if self.env.MAP[y][x]!=0:
						p = sess.run(
									self.PGNetwork.pi, 
									feed_dict={
										self.PGNetwork.inputs: [self.env.cv_state_onehot[x+(y-1)*self.env.bounds_x[1]-1]],
										self.PGNetwork.inputt: [self.env.cv_task_onehot[task]]
									})
					
						current_policy[x,y,task] = p.ravel().tolist()
						
		
		if epoch % self.plot_model == 0 or epoch==1:
			self.plot_figure.plot(current_policy, epoch)
						
		return current_policy

	

	def _process_experience_normalize(self, sess, states, tasks, actions, drewards, current_policy):

		batch_ss, batch_ts, batch_as, batch_drs = [], [], [], []   
		share_ss, share_ts, share_as, share_drs = [], [], [], []   
		samples = {}
		action_samples = {}

		# break trajectories to samples and put to dictionaries:
		# samples[state,task] = discounted_rewards
		# action_samples = {}

		for i in range(self.num_task):
			for index, state in enumerate(states[i]):
				
				if (state[0], state[1], tasks[i][index]) not in samples:
					samples[state[0], state[1], tasks[i][index]] = []
					action_samples[state[0], state[1], tasks[i][index]] = []
				
				samples[state[0], state[1], tasks[i][index]].append(drewards[i][index])
				action_samples[state[0], state[1], tasks[i][index]].append(actions[i][index])
		
		# get samples from dictionaries and build trainning batch			
		for v in samples.keys():
			state_index = v[0]-1+(v[1]-1)*self.env.bounds_x[1]

			# normalize discounted rewards
			if abs(np.std(samples[v]))>1e-3:
				samples[v] = (np.array(samples[v])-np.mean(samples[v]))/np.std(samples[v])
			
			for i, reward in enumerate(samples[v]):

				# original samples
				batch_ss.append(self.env.cv_state_onehot[state_index])
				batch_ts.append(self.env.cv_task_onehot[v[2]])
				batch_as.append(self.env.cv_action_onehot[action_samples[v][i]])
				batch_drs.append(reward)

				# interpolate sharing samples
				if self.share_exp:
					# only interpolate samples in sharing areas 
					if self.env.MAP[v[1]][v[0]]==2:
						share_ss.append(self.env.cv_state_onehot[state_index])
						share_ts.append(self.env.cv_task_onehot[1-v[2]]) #task_1 => task_0 and task_0 => task_1
						share_as.append(self.env.cv_action_onehot[action_samples[v][i]])
						important_weight = current_policy[v[0],v[1],1-v[2]][action_samples[v][i]]/current_policy[v[0],v[1],v[2]][action_samples[v][i]]
						share_drs.append(important_weight*reward)
					# keep samples in non-sharing areas to avoid bias (not sure now, need read and experiment to make final decision)
					else:
						share_ss.append(self.env.cv_state_onehot[state_index])
						share_ts.append(self.env.cv_task_onehot[v[2]])
						share_as.append(self.env.cv_action_onehot[action_samples[v][i]])
						share_drs.append(reward)
				
		return batch_ss, batch_ts, batch_as, batch_drs, share_ss, share_ts, share_as, share_drs, samples
	


	def _make_batch(self, sess, epoch):
		current_policy = self._prepare_current_policy(sess, epoch)

		# states = [
		#   task1		[[---episode_1---],...,[---episode_n---]],
		#   task2		[[---episode_1---],...,[---episode_n---]]
		#		   ]

		states, tasks, actions, rewards = self.rollout.rollout_batch(sess, self.PGNetwork, current_policy, epoch)     

		discounted_rewards = [[],[]]
		for index, task_rewards in enumerate(rewards):
			for ep_reward in task_rewards:
				discounted_rewards[index]+=self._discount_rewards(ep_reward)

		states[0] = np.concatenate(states[0])
		tasks[0] = np.concatenate(tasks[0])     
		actions[0] = np.concatenate(actions[0])     
		rewards[0] = np.concatenate(rewards[0])

		if self.num_task > 1:       
			states[1] = np.concatenate(states[1])
			tasks[1] = np.concatenate(tasks[1])
			actions[1] = np.concatenate(actions[1])
			rewards[1] = np.concatenate(rewards[1])
		
	
		batch_ss, batch_ts, batch_as, batch_drs, share_ss, share_ts, share_as, share_drs, samples = self._process_experience_normalize(sess, states, tasks, actions, discounted_rewards, current_policy) 

		return share_ss, share_ts, share_as, share_drs, batch_ss, batch_ts, batch_as, batch_drs, np.concatenate(rewards)
		
		
	def train(self, sess, saver):
		# run with saved initial model
		epoch = 0
		num_sample = 0
		while epoch < self.num_epochs:
			print('[TRAINING {}] epoch {}/{}'.format(self.save_name, epoch, self.num_epochs - 1), end = '\r', flush = True)
			
			# ROLLOUT SAMPLE
			#---------------------------------------------------------------------------------------------------------------------#	
			share_ss, share_ts, share_as, share_drs, states_mb, tasks_mb, actions_mb, discounted_rewards_mb, rewards_mb  = self._make_batch(sess, epoch)
			#---------------------------------------------------------------------------------------------------------------------#	
			

			# UPDATE NETWORK
			#---------------------------------------------------------------------------------------------------------------------#	
			
			# base_line
			gradients = sess.run([self.PGNetwork.gvs], feed_dict={
																self.PGNetwork.inputs: states_mb,
																self.PGNetwork.inputt: tasks_mb,
																self.PGNetwork.actions: actions_mb,
																self.PGNetwork.rewards: discounted_rewards_mb 
																})
			if self.share_exp:
				# combine gradient
				if self.combine_gradent:
					gradients_share = sess.run([self.PGNetwork.gvs], feed_dict={
																self.PGNetwork.inputs: share_ss,
																self.PGNetwork.inputt: share_ts,
																self.PGNetwork.actions: share_as,
																self.PGNetwork.rewards: share_drs 
																})	

					# final_grad = weight*grad+(1-weight)*grad_share 
					# gradients = [[(array_grad_1, array_var_1), (array_grad_2, array_var_2), (array_grad_3, array_var_3), (array_grad_4, array_var_4)]]

					for i, grad in enumerate(gradients[0]):
						# print(gradients[0][i][0].shape, gradients_share[0][i][0].shape)
						gradients[0][i] = (self.share_exp_weight * gradients[0][i][0] + (1 - self.share_exp_weight) * gradients_share[0][i][0], gradients[0][i][1])

				# combine sample
				else: 
					gradients = sess.run([self.PGNetwork.gvs], feed_dict={
																self.PGNetwork.inputs: share_ss+states_mb,
																self.PGNetwork.inputt: share_ts+tasks_mb,
																self.PGNetwork.actions: share_as+actions_mb,
																self.PGNetwork.rewards: share_drs+discounted_rewards_mb 
																})	
				
			
			# update network weight from computed gradient	
			feed_dict = {}
			for i, grad in enumerate(gradients[0]):
			    feed_dict[self.PGNetwork.placeholder_gradients[i][0]] = grad[0]
			_ = sess.run([self.PGNetwork.train_opt], feed_dict=feed_dict)
			#---------------------------------------------------------------------------------------------------------------------#	
			


			# WRITE TF SUMMARIES
			#---------------------------------------------------------------------------------------------------------------------#	
			num_sample+=len(rewards_mb)
			total_reward_of_that_batch = np.sum(rewards_mb)
			mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, self.num_episode)
			summary = sess.run(self.write_op, feed_dict={self.PGNetwork.mean_reward: mean_reward_of_that_batch})

			self.writer.add_summary(summary, num_sample)
			self.writer.flush()
			#---------------------------------------------------------------------------------------------------------------------#	


			# SAVE MODEL
			#---------------------------------------------------------------------------------------------------------------------#	
			if epoch % self.save_model == 0:
				saver.save(sess, 'checkpoints/' + self.save_name + '.ckpt')
			#---------------------------------------------------------------------------------------------------------------------#		
			epoch += 1  