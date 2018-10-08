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
			policies,
			writer,
			write_op,
			action_size,
			num_task,
			num_iters,			
			gamma,
			plot_model,
			save_model,
			save_name,
			num_episode,
			share_exp,
			immortal,
			combine_gradent,
			share_exp_weight,
			sort_init,
			use_laser,
			timer
			):

		self.map_index = map_index
		self.PGNetwork = policies

		self.writer = writer
		self.write_op = write_op
		self.action_size = action_size
		self.num_task = num_task
		self.num_iters = num_iters
		self.gamma = gamma
		self.save_name = save_name
		self.plot_model = plot_model
		self.save_model = save_model
		self.immortal = immortal

		self.gradients = [[] for i in range(self.num_task)]
		self.batch_eps = [[] for i in range(self.num_task)]

		self.num_episode =  num_episode
		self.combine_gradent = combine_gradent
		self.share_exp = share_exp
		self.share_exp_weight = share_exp_weight

		self.env = Terrain(self.map_index, use_laser, immortal)

		assert self.num_task <= self.env.num_task

		self.plot_figure = PlotFigure(self.save_name, self.env, self.num_task, os.path.join('plot', timer))

		self.rollout = Rollout(number_episode = self.num_episode, num_task = self.num_task, map_index = self.map_index, sort_init = sort_init, use_laser = use_laser, immortal = immortal)


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
		
		for task in range(self.num_task):
			for (x, y) in self.env.state_space:
				state_index = self.env.state_to_index[y][x]
				p = sess.run(
							self.PGNetwork[task].pi, 
							feed_dict={
								self.PGNetwork[task].inputs: [self.env.cv_state_onehot[state_index]],
							})
			
				current_policy[x,y,task] = p.ravel().tolist()
						
		
		if (epoch+1) % self.plot_model == 0 or epoch == 0:
			self.plot_figure.plot(current_policy, epoch + 1, self.gradients, self.batch_eps)
			self.gradients = [[] for i in range(self.num_task)]
			self.batch_eps = [[] for i in range(self.num_task)]
						
		return current_policy

	

	def _process_experience_normalize(self, sess, states, tasks, actions, drewards, current_policy):
		make_holder = lambda x: [[] for i in range(x)]

		batch_ss, batch_as, batch_drs = [make_holder(self.num_task) for i in range(3)]
		share_ss, share_as, share_drs = [make_holder(self.num_task) for i in range(3)]
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
			state_index = self.env.state_to_index[v[1]][v[0]]

			# normalize discounted rewards
			# if abs(np.std(samples[v]))>1e-3:
			# 	samples[v] = (np.array(samples[v])-np.mean(samples[v]))/np.std(samples[v])
			
			for i, (reward, action) in enumerate(zip(samples[v], action_samples[v])):

				# original samples
				batch_ss[v[2]].append(self.env.cv_state_onehot[state_index])
				batch_as[v[2]].append(self.env.cv_action_onehot[action_samples[v][i]])

				# interpolate sharing samples only interpolate samples in sharing areas 
				if self.share_exp and self.env.MAP[v[1]][v[0]]==2:
					fx = (current_policy[v[0],v[1],1-v[2]][action]+current_policy[v[0],v[1],v[2]][action])/2
					batch_drs[v[2]].append(current_policy[v[0],v[1],v[2]][action] * reward / fx)

					share_ss[1-v[2]].append(self.env.cv_state_onehot[state_index])
					share_as[1-v[2]].append(self.env.cv_action_onehot[action_samples[v][i]])

					share_drs[1-v[2]].append(current_policy[v[0],v[1],1-v[2]][action] * reward / fx)
				
				else:
					batch_drs[v[2]].append(reward)

		return batch_ss, batch_as, batch_drs, share_ss, share_as, share_drs, samples
	


	def _make_batch(self, sess, epoch):
		current_policy = self._prepare_current_policy(sess, epoch)

		# states = [
		#   task1		[[---episode_1---],...,[---episode_n---]],
		#   task2		[[---episode_1---],...,[---episode_n---]],
		#   .
		#   .
		#	task_k      [[---episode_1---],...,[---episode_n---]],
		#		   ]

		states, tasks, actions, rewards = self.rollout.rollout_batch(sess, self.PGNetwork, current_policy, epoch)     
		for task, task_states in enumerate(states):
			self.batch_eps[task] = task_states

		discounted_rewards = [[] for i in range(self.num_task)]
		for index, task_rewards in enumerate(rewards):
			for ep_reward in task_rewards:
				discounted_rewards[index]+=self._discount_rewards(ep_reward)

		for i in range(self.num_task):
			states[i] = np.concatenate(states[i])
			tasks[i] = np.concatenate(tasks[i])
			actions[i] = np.concatenate(actions[i])
			rewards[i] = np.concatenate(rewards[i])
		
	
		batch_ss, batch_as, batch_drs, share_ss, share_as, share_drs, samples = self._process_experience_normalize(sess, states, tasks, actions, discounted_rewards, current_policy) 

		return share_ss, share_as, share_drs, batch_ss, batch_as, batch_drs, rewards
		
		
	def train(self, sess, saver):
		# run with saved initial model
		epoch = 0
		num_sample = 0
		while num_sample < self.num_iters:
			print('[TRAINING {}] state_size {}, action_size {}, epoch {}'.format(self.save_name, self.env.cv_state_onehot.shape[1], self.action_size, epoch), end = '\r', flush = True)
			
			# ROLLOUT SAMPLE
			#---------------------------------------------------------------------------------------------------------------------#	
			share_ss, share_as, share_drs, states_mb, actions_mb, discounted_rewards_mb, rewards_mb  = self._make_batch(sess, epoch)
			#---------------------------------------------------------------------------------------------------------------------#	
			

			# UPDATE NETWORK
			#---------------------------------------------------------------------------------------------------------------------#	
			
			if self.share_exp:
				# combine gradient
				if self.combine_gradent:
					for task_index in range(self.num_task):
						gradients 		= sess.run([self.PGNetwork[task_index].gvs], feed_dict={
																		self.PGNetwork[task_index].inputs: states_mb[task_index],
																		self.PGNetwork[task_index].actions: actions_mb[task_index],
																		self.PGNetwork[task_index].rewards: discounted_rewards_mb[task_index]
																		})

						if len(share_ss[task_index]) > 0:
							gradients_share = sess.run([self.PGNetwork[task_index].gvs], feed_dict={
																			self.PGNetwork[task_index].inputs: share_ss[task_index],
																			self.PGNetwork[task_index].actions: share_as[task_index],
																			self.PGNetwork[task_index].rewards: share_drs[task_index] 
																			})	

							# final_grad = weight*grad+(1-weight)*grad_share 
							# gradients = [[(array_grad_1, array_var_1), (array_grad_2, array_var_2), (array_grad_3, array_var_3), (array_grad_4, array_var_4)]]
							for i, grad in enumerate(gradients[0]):
								gradients[0][i] = (gradients[0][i][0] + gradients_share[0][i][0], gradients[0][i][1])
								# gradients[0][i] = (self.share_exp_weight * gradients[0][i][0] + (1 - self.share_exp_weight) * gradients_share[0][i][0], gradients[0][i][1])

						#update network weight from computed gradient	
						feed_dict = {}
						for i, grad in enumerate(gradients[0]):
						    feed_dict[self.PGNetwork[task_index].placeholder_gradients[i][0]] = grad[0]
						_ = sess.run([self.PGNetwork[task_index].train_opt], feed_dict=feed_dict)

						for grad in gradients[0]:
							self.gradients[task_index].append(np.sum(grad[0]))

				# combine sample
				else: 
					for task_index in range(self.num_task):
						gradients = sess.run([self.PGNetwork[task_index].gvs], feed_dict={
																	self.PGNetwork[task_index].inputs: share_ss[task_index] + states_mb[task_index],
																	self.PGNetwork[task_index].actions: share_as[task_index] + actions_mb[task_index],
																	self.PGNetwork[task_index].rewards: share_drs[task_index] + discounted_rewards_mb[task_index] 
																	})	
						#update network weight from computed gradient	
						feed_dict = {}
						for i, grad in enumerate(gradients[0]):
						    feed_dict[self.PGNetwork[task_index].placeholder_gradients[i][0]] = grad[0]
						_ = sess.run([self.PGNetwork[task_index].train_opt],feed_dict=feed_dict)
						
						for grad in gradients[0]:
							self.gradients[task_index].append(np.sum(grad[0]))
			else:
				for task_index in range(self.num_task):
					gradients = sess.run([self.PGNetwork[task_index].gvs], feed_dict={
																		self.PGNetwork[task_index].inputs: states_mb[task_index],
																		self.PGNetwork[task_index].actions: actions_mb[task_index],
																		self.PGNetwork[task_index].rewards: discounted_rewards_mb[task_index] 
																		})
					#update network weight from computed gradient	
					feed_dict = {}
					for i, grad in enumerate(gradients[0]):
					    feed_dict[self.PGNetwork[task_index].placeholder_gradients[i][0]] = grad[0]
					_ = sess.run([self.PGNetwork[task_index].train_opt],feed_dict=feed_dict)

					for grad in gradients[0]:
							self.gradients[task_index].append(np.sum(grad[0]))

			#---------------------------------------------------------------------------------------------------------------------#	
			

			# WRITE TF SUMMARIES
			#---------------------------------------------------------------------------------------------------------------------#	
			sum_dict = {}
			for i in range(self.num_task):
				sum_dict[self.PGNetwork[i].mean_reward] = np.mean(rewards_mb[i])
				num_sample += rewards_mb[i].shape[0]

			summary = sess.run(self.write_op, feed_dict = sum_dict)

			self.writer.add_summary(summary, num_sample)
			self.writer.flush()
			#---------------------------------------------------------------------------------------------------------------------#	


			# SAVE MODEL
			#---------------------------------------------------------------------------------------------------------------------#	
			if epoch % self.save_model == 0:
				saver.save(sess, 'checkpoints/' + self.save_name + '.ckpt')
			#---------------------------------------------------------------------------------------------------------------------#		
			epoch += 1  