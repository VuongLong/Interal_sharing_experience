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
						
		
		if epoch%self.plot_model==0 or epoch==1:
			self.plot_figure.plot(current_policy, epoch)
						
		return current_policy

	

	def _process_experience_normailize(self, sess, states, actions, drewards, current_policy):

		batch_ss, batch_as, batch_drs = [[],[]], [[],[]], [[],[]]
		share_ss, share_as, share_drs = [[],[]], [[],[]], [[],[]]
		state_dict = {}
		action_dict = {}

		for task in range(len(states)):
			for index, state in enumerate(states[task]):
				
				if state_dict.get((state[0], state[1], task),-1)==-1:
					state_dict[state[0], state[1], task]=[]
					action_dict[state[0], state[1], task]=[]
				
				state_dict[state[0], state[1], task].append(drewards[task][index])
				action_dict[state[0], state[1], task].append(actions[task][index])
		
		for v in state_dict.keys():
			state_index = v[0]+(v[1]-1)*self.env.bounds_x[1]-1
			#normalize discounted rewards
			#if abs(np.std(state_dict[v]))>1e-3:
			#	state_dict[v] = (np.array(state_dict[v])-np.mean(state_dict[v]))/np.std(state_dict[v])
			
			for i, (action, reward) in enumerate(zip(action_dict[v], state_dict[v])):
				if self.share_exp:
					fx = (current_policy[v[0],v[1],1-v[2]][action]+current_policy[v[0],v[1],v[2]][action])/2

					batch_ss[v[2]].append(self.env.cv_state_onehot[state_index])
					batch_as[v[2]].append(self.env.cv_action_onehot[action])
					if self.env.MAP[v[1]][v[0]]==2:
						important_weight = current_policy[v[0],v[1],v[2]][action]/fx
					else:
						important_weight = 1.0	
					batch_drs[v[2]].append(important_weight*reward)
					
					if self.env.MAP[v[1]][v[0]]==2:
						share_ss[1-v[2]].append(self.env.cv_state_onehot[state_index])
						share_as[1-v[2]].append(self.env.cv_action_onehot[action])
						important_weight = current_policy[v[0],v[1],1-v[2]][action]/fx
						share_drs[1-v[2]].append(important_weight*reward)
				else:
					batch_ss[v[2]].append(self.env.cv_state_onehot[state_index])
					batch_as[v[2]].append(self.env.cv_action_onehot[action])
					batch_drs[v[2]].append(reward)
				
		return batch_ss, batch_as, batch_drs, share_ss, share_as, share_drs
	


	def _make_batch(self, sess, epoch):
		current_policy = self._prepare_current_policy(sess, epoch)

		# states = [
		#task1		[[---episode_1---],...,[---episode_n---]],
		#task2		[[---episode_1---],...,[---episode_n---]]
		#			]
		states, tasks, actions, rewards = self.rollout.rollout_batch(sess, self.PGNetwork, current_policy, epoch)     

		discounted_rewards = [[],[]]
		for index, task_rewards in enumerate(rewards):
			for i, ep_reward in enumerate(task_rewards):
				discounted_rewards[index]+=self._discount_rewards(ep_reward)
				#if ep_reward[-1] == 1:
				#	print  states[index][i][-1], tasks[index][i][-1]

		states[0] = np.concatenate(states[0])       
		states[1] = np.concatenate(states[1])
		tasks[0] = np.concatenate(tasks[0])     
		tasks[1] = np.concatenate(tasks[1])
		actions[0] = np.concatenate(actions[0])     
		actions[1] = np.concatenate(actions[1])
		rewards[0] = np.concatenate(rewards[0])
		rewards[1] = np.concatenate(rewards[1])
		
	
		batch_ss, batch_as, batch_drs, share_ss, share_as, share_drs= self._process_experience_normailize(sess, states, actions, discounted_rewards, current_policy) 

		return share_ss, share_as, share_drs, batch_ss, batch_as, batch_drs, np.concatenate(rewards)
		
		
	def train(self, sess, saver):
		#run with saved initial model
		epoch = 1
		num_sample = 0
		while epoch < self.num_epochs + 1:
			print epoch
			
			#---------------------------------------------------------------------------------------------------------------------#	
			#ROLOUT state_dict
			share_ss, share_as, share_drs, states_mb, actions_mb, discounted_rewards_mb, rewards_mb  = self._make_batch(sess, epoch)
			#---------------------------------------------------------------------------------------------------------------------#	
		
			#---------------------------------------------------------------------------------------------------------------------#	
			#UPDATE NETWORK
			
			test_num_task = 2
			if self.share_exp:
				if self.combine_gradent: #combine gradient
					#base_line
					for task_index in range(test_num_task):
						gradients = sess.run([self.PGNetwork[task_index].gvs], feed_dict={
																		self.PGNetwork[task_index].inputs: states_mb[task_index],
																		self.PGNetwork[task_index].actions: actions_mb[task_index],
																		self.PGNetwork[task_index].rewards: discounted_rewards_mb[task_index]
																		})
						if len(share_ss[task_index])>0:
							gradients_share = sess.run([self.PGNetwork[task_index].gvs], feed_dict={
																	self.PGNetwork[task_index].inputs: share_ss[task_index],
																	self.PGNetwork[task_index].actions: share_as[task_index],
																	self.PGNetwork[task_index].rewards: share_drs[task_index] 
																	})	
							#final_grad = weight*grad+(1-weight===)*grad_share 
							for i, grad in enumerate(gradients[0]):
								for g,v  in enumerate(grad[0]):
									#gradients[0][i]=(self.share_exp_weight*gradients[0][i][0]+(1-self.share_exp_weight)*gradients_share[0][i][0],gradients[0][i][1])
									gradients[0][i]=(gradients[0][i][0]+gradients_share[0][i][0],gradients[0][i][1])
							
						#update network weight from computed gradient	
						feed_dict = {}
						for i, grad in enumerate(gradients[0]):
						    feed_dict[self.PGNetwork[task_index].placeholder_gradients[i][0]] = grad[0]
						_ = sess.run([self.PGNetwork[task_index].train_opt],feed_dict=feed_dict)
				else: #combine state_dict
					for task_index in range(test_num_task):
						gradients = sess.run([self.PGNetwork[task_index].gvs], feed_dict={
																	self.PGNetwork[task_index].inputs: share_ss[task_index]+states_mb[task_index],
																	self.PGNetwork[task_index].actions: share_as[task_index]+actions_mb[task_index],
																	self.PGNetwork[task_index].rewards: share_drs[task_index]+discounted_rewards_mb[task_index] 
																	})	
						#update network weight from computed gradient	
						feed_dict = {}
						for i, grad in enumerate(gradients[0]):
						    feed_dict[self.PGNetwork[task_index].placeholder_gradients[i][0]] = grad[0]
						_ = sess.run([self.PGNetwork[task_index].train_opt],feed_dict=feed_dict)
						'''
						
						sess.run([self.PGNetwork[task_index].inter_train_opt], feed_dict={
																	self.PGNetwork[task_index].inputs: share_ss[task_index]+states_mb[task_index],
																	self.PGNetwork[task_index].actions: share_as[task_index]+actions_mb[task_index],
																	self.PGNetwork[task_index].rewards: share_drs[task_index]+discounted_rewards_mb[task_index] 
																	})	
						'''

			else:
				for task_index in range(test_num_task):
					#base_line
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
					'''
					sess.run([self.PGNetwork[task_index].inter_train_opt], feed_dict={
																		self.PGNetwork[task_index].inputs: states_mb[task_index],
																		self.PGNetwork[task_index].actions: actions_mb[task_index],
																		self.PGNetwork[task_index].rewards: discounted_rewards_mb[task_index]
																		})
					'''

			#---------------------------------------------------------------------------------------------------------------------#	
			


			#---------------------------------------------------------------------------------------------------------------------#	
			# WRITE TF SUMMARIES
			num_sample+=len(rewards_mb)

			total_reward_of_that_batch = np.sum(rewards_mb)
			mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, self.num_episide)
			summary = sess.run(self.write_op, feed_dict={
												self.PGNetwork[0].mean_reward: mean_reward_of_that_batch
												})

			#self.writer.add_summary(summary, num_sample)
			self.writer.add_summary(summary, epoch)
			self.writer.flush()
			#---------------------------------------------------------------------------------------------------------------------#	


			#---------------------------------------------------------------------------------------------------------------------#	
			# SAVE MODEL
			if epoch % self.save_model == 0:
				saver.save(sess, '../models/'+self.save_name+'.ckpt')
				print("Model saved")
			#---------------------------------------------------------------------------------------------------------------------#		
			epoch += 1  