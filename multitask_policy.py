# -*- coding: utf-8 -*-

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import random                # Handling random number generation
import time                  # Handling time calculation
import math
from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs
import copy
import threading
from rollout import Rollout
from env.map import ENV_MAP
from plot_figure import PlotFigure
import os
from env.terrain import Terrain

class MultitaskPolicy(object):

	def __init__(
			self,
			map_index,
			policy,
			value,
			oracle,
			writer,
			write_op,
			num_epochs,			
			gamma,
			plot_model,
			save_model,
			save_name,

			num_episide,
			share_exp,
			combine_gradent,
			share_exp_weight
			):

		self.map_index = map_index
		self.PGNetwork = policy
		self.VNetwork = value
		self.ZNetwork = oracle

		self.writer = writer
		self.write_op = write_op
		self.num_epochs = num_epochs
		self.gamma = gamma
		self.save_name = save_name
		self.plot_model = plot_model
		self.save_model = save_model

		self.num_episide =  num_episide
		self.combine_gradent = combine_gradent
		self.share_exp = share_exp
		self.share_exp_weight = share_exp_weight

		self.env = Terrain(self.map_index)

		self.plot_figure = PlotFigure(self.save_name, self.env)

		self.rollout = Rollout(number_episode = self.num_episide, map_index = self.map_index)


	def _discount_rewards(self, episode_rewards, episode_states, episode_nexts, task, current_value):
		discounted_episode_rewards = np.zeros_like(episode_rewards)
		next_value = 0.0
		if episode_rewards[-1] == 1:
			next_value = 0.0
		else:
			next_value = current_value[episode_nexts[-1][0],episode_nexts[-1][1], task]

		for i in reversed(range(len(episode_rewards))):
			next_value = episode_rewards[i] + self.gamma * next_value  
			discounted_episode_rewards[i] = next_value

		return discounted_episode_rewards.tolist()

	def _GAE(self, episode_rewards, episode_states, episode_nexts, task, current_value):
		ep_GAE = np.zeros_like(episode_rewards)
		TD_error = np.zeros_like(episode_rewards)
		lamda=0.96

		next_value = 0.0
		if episode_rewards[-1] == 1:
			next_value = 0.0
		else:
			next_value = current_value[episode_nexts[-1][0],episode_nexts[-1][1], task]

		for i in reversed(range(len(episode_rewards))):
			TD_error[i] = episode_rewards[i]+self.gamma*next_value-current_value[episode_states[i][0],episode_states[i][1], task]
			next_value = current_value[episode_states[i][0],episode_states[i][1], task]

		ep_GAE[len(episode_rewards)-1] = TD_error[len(episode_rewards)-1]
		weight = self.gamma*lamda
		for i in reversed(range(len(episode_rewards)-1)):
			ep_GAE[i] += TD_error[i]+weight*ep_GAE[i+1]

		return ep_GAE.tolist()	


	def _prepare_current_policy(self, sess, epoch):
		current_policy = {}
		current_value = {}
		current_Z = {}
		range_x = [0, self.env.bounds_x[1]+1]
		range_y = [0, self.env.bounds_y[1]+1]

		for task in range(self.env.num_task):
			for x in range(range_x[0]+1,range_x[1]):
				for y in range(range_y[0]+1,range_y[1]):
					if self.env.MAP[y][x]!=0:
						p = sess.run(
									self.PGNetwork[task].pi, 
									feed_dict={
										self.PGNetwork[task].inputs: [self.env.cv_state_onehot[x+(y-1)*self.env.bounds_x[1]-1]]})
					
						current_policy[x,y,task] = p.ravel().tolist()
						v = sess.run(
									self.VNetwork[task].value, 
									feed_dict={
										self.VNetwork[task].inputs: [self.env.cv_state_onehot[x+(y-1)*self.env.bounds_x[1]-1]]})
						current_value[x,y,task] = v[0][0]
						

		if epoch%self.plot_model==0 or epoch==1:
			self.plot_figure.plot(current_policy, epoch)
						
		return current_policy, current_value

	

	def _process_experience_normailize(self, epoch, states, actions, drewards, GAEs, next_states, current_policy, current_value):

		batch_ss, batch_as, batch_Qs, batch_Ts = [[],[]], [[],[]], [[],[]], [[],[]]
		share_ss, share_as, share_Qs, share_Ts = [[],[]], [[],[]], [[],[]], [[],[]]

		for task in range(len(states)):
			for i, (state, action, actual_value, gae, next_state) in enumerate(zip(states[task], actions[task], drewards[task],GAEs[task], next_states[task])):	
				state_index = state[0]+(state[1]-1)*self.env.bounds_x[1]-1

				if self.share_exp:
					fx = (current_policy[state[0],state[1],task][action]+current_policy[state[0],state[1],1-task][action])/2
					if self.env.MAP[state[1]][state[0]]==2:
						important_weight = current_policy[state[0],state[1],task][action]/fx
					else:
						important_weight = 1.0	
					
					advantage = actual_value - current_value[state[0],state[1],task]

					batch_ss[task].append(self.env.cv_state_onehot[state_index])
					batch_as[task].append(self.env.cv_action_onehot[action])
					batch_Qs[task].append(actual_value)
					batch_Ts[task].append(important_weight*gae)


					if self.env.MAP[state[1]][state[0]]==2:
						share_ss[1-task].append(self.env.cv_state_onehot[state_index])
						share_as[1-task].append(self.env.cv_action_onehot[action])
						important_weight = current_policy[state[0],state[1],1-task][action]/fx
						share_Ts[1-task].append(important_weight*gae)
				else:

					advantage = actual_value - current_value[state[0],state[1],task]
					
					batch_ss[task].append(self.env.cv_state_onehot[state_index])
					batch_as[task].append(self.env.cv_action_onehot[action])
					batch_Qs[task].append(actual_value)
					batch_Ts[task].append(gae)


				
		return batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts
	

	def _make_batch(self, sess, epoch):
		current_policy, current_value= self._prepare_current_policy(sess, epoch)

		# states = [
		#task1		[[---episode_1---],...,[---episode_n---]],
		#task2		[[---episode_1---],...,[---episode_n---]]
		#			]
		states, tasks, actions, rewards, next_states= self.rollout.rollout_batch(sess, self.PGNetwork, current_policy, epoch)     

		discounted_rewards = [[],[]]
		GAEs = [[],[]]
		for task in range(len(rewards)):
			for i, (ep_state, ep_next, ep_reward) in enumerate(zip(states[task], next_states[task], rewards[task])):	
				discounted_rewards[task]+=self._discount_rewards(ep_reward, ep_state, ep_next, task, current_value)
				GAEs[task]+=self._GAE(ep_reward, ep_state, ep_next, task, current_value)

		states[0] = np.concatenate(states[0])       
		states[1] = np.concatenate(states[1])
		tasks[0] = np.concatenate(tasks[0])     
		tasks[1] = np.concatenate(tasks[1])
		actions[0] = np.concatenate(actions[0])     
		actions[1] = np.concatenate(actions[1])
		rewards[0] = np.concatenate(rewards[0])
		rewards[1] = np.concatenate(rewards[1])
		next_states[0] = np.concatenate(next_states[0])
		next_states[1] = np.concatenate(next_states[1])
		
		
		batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts = self._process_experience_normailize(epoch, states, actions, discounted_rewards, GAEs, next_states, current_policy, current_value) 

		return batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts, np.concatenate(rewards)
		
		
	def train(self, sess, saver):
		#run with saved initial model
		epoch = 1
		num_sample = 0
		while epoch < self.num_epochs + 1:
			print epoch
			
			#---------------------------------------------------------------------------------------------------------------------#	
			#ROLOUT state_dict
			batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts, rewards_mb  = self._make_batch(sess, epoch)
			#---------------------------------------------------------------------------------------------------------------------#	
		
			#---------------------------------------------------------------------------------------------------------------------#	
			#UPDATE NETWORK
			
			test_num_task = 2
			if self.share_exp:
				
				for task_index in range(test_num_task):
					
					
					sess.run([self.VNetwork[task_index].train_opt], feed_dict={
																self.VNetwork[task_index].inputs: batch_ss[task_index],
																self.VNetwork[task_index].rewards: batch_Qs[task_index] 
																})	
					share_ss[task_index]+=batch_ss[task_index]
					share_as[task_index]+=batch_as[task_index]
					share_Ts[task_index]+=batch_Ts[task_index]
					#normalize discounted rewards
					#share_Ts[task_index] = (np.array(share_Ts[task_index])-np.mean(share_Ts[task_index]))/np.std(share_Ts[task_index])
					#share_Ts[task_index].tolist()

					sess.run([self.PGNetwork[task_index].train_opt], feed_dict={
																self.PGNetwork[task_index].inputs: share_ss[task_index],
																self.PGNetwork[task_index].actions: share_as[task_index],
																self.PGNetwork[task_index].rewards: share_Ts[task_index]
																})	
					

			else:
				for task_index in range(test_num_task):
					#base_line
					loss,_ =sess.run([self.VNetwork[task_index].loss, self.VNetwork[task_index].train_opt], feed_dict={
																self.VNetwork[task_index].inputs: batch_ss[task_index],
																self.VNetwork[task_index].rewards: batch_Qs[task_index] 
																})	
					#batch_Ts[task_index] = (np.array(batch_Ts[task_index])-np.mean(batch_Ts[task_index]))/np.std(batch_Ts[task_index])
					#batch_Ts[task_index].tolist()
					sess.run([self.PGNetwork[task_index].train_opt], feed_dict={
																		self.PGNetwork[task_index].inputs: batch_ss[task_index],
																		self.PGNetwork[task_index].actions: batch_as[task_index],
																		self.PGNetwork[task_index].rewards: batch_Ts[task_index]
																		})
					
			#---------------------------------------------------------------------------------------------------------------------#	
			


			#---------------------------------------------------------------------------------------------------------------------#	
			# WRITE TF SUMMARIES
			num_sample+=len(rewards_mb)

			total_reward_of_that_batch = np.sum(rewards_mb)
			mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, self.num_episide)
			summary = sess.run(self.write_op, feed_dict={
												self.PGNetwork[0].mean_reward: mean_reward_of_that_batch
												})

			self.writer.add_summary(summary, num_sample)
			#self.writer.add_summary(summary, epoch)
			self.writer.flush()
			#---------------------------------------------------------------------------------------------------------------------#	


			#---------------------------------------------------------------------------------------------------------------------#	
			# SAVE MODEL
			if epoch % self.save_model == 0:
				saver.save(sess, '../models/'+self.save_name+'.ckpt')
				print("Model saved")
			#---------------------------------------------------------------------------------------------------------------------#		
			epoch += 1  