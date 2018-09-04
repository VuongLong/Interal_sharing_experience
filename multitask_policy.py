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
			state_size,
			action_size,
			task_size,
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
		self.state_size = state_size 
		self.action_size = action_size
		self.task_size = task_size
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
									self.PGNetwork.pi, 
									feed_dict={
										self.PGNetwork.inputs: [self.env.cv_state_onehot[x+(y-1)*self.env.bounds_x[1]-1]],
										self.PGNetwork.inputt: [self.env.cv_task_onehot[task]]})
					
						current_policy[x,y,task] = p.ravel().tolist()
						
		
		if epoch%self.plot_model==0 or epoch==1:
			self.plot_figure.plot(current_policy, epoch)
						
		return current_policy

	

	def _process_experience_normailize(self, sess, states, tasks, actions, drewards, current_policy):

		batch_ss, batch_ts, batch_as, batch_drs = [], [], [], []   
		share_ss, share_ts, share_as, share_drs = [], [], [], []   
		sample = {}
		action_sample = {}

		# break trajecties to samples and put to dictionarie:
		# sample[state,task]=discounted_reward
		# sample_action = action

		for i in range(len(states)):
			for index, state in enumerate(states[i]):
				
				if sample.get((states[i][index][0], states[i][index][1], tasks[i][index]),-10000)==-10000:
					sample[states[i][index][0], states[i][index][1], tasks[i][index]]=[]
					action_sample[states[i][index][0], states[i][index][1], tasks[i][index]]=[]
				
				sample[states[i][index][0], states[i][index][1], tasks[i][index]].append(drewards[i][index])
				action_sample[states[i][index][0], states[i][index][1], tasks[i][index]].append(actions[i][index])
		
		# get sample from dictionaries and build trainning batch			
		for v in sample.keys():
			state_index = v[0]+(v[1]-1)*self.env.bounds_x[1]-1

			#normalize discounted rewards
			if abs(np.std(sample[v]))>1e-3:
				sample[v] = (np.array(sample[v])-np.mean(sample[v]))/np.std(sample[v])
			
			for i, reward in enumerate(sample[v]):

				# original samples
				batch_ss.append(self.env.cv_state_onehot[state_index])

				batch_ts.append(self.env.cv_task_onehot[v[2]])
				batch_as.append(self.env.cv_action_onehot[action_sample[v][i]])
				batch_drs.append(reward)

				# interpolate sharing sample
				if self.share_exp:
					#only interpolate sample in sharing areas 
					if self.env.MAP[v[1]][v[0]]==2:
						share_ss.append(self.env.cv_state_onehot[state_index])
						share_ts.append(self.env.cv_task_onehot[1-v[2]]) #task_1 => task_0 and task_0 => task_1
						share_as.append(self.env.cv_action_onehot[action_sample[v][i]])
						important_weight = current_policy[v[0],v[1],1-v[2]][action_sample[v][i]]/current_policy[v[0],v[1],v[2]][action_sample[v][i]]
						share_drs.append(important_weight*reward)
					# keep sample in non-sharing areas to avoid bias (not sure now, need read and experiment to make final decision)
					else:
						share_ss.append(self.env.cv_state_onehot[state_index])
						share_ts.append(self.env.cv_task_onehot[v[2]])
						share_as.append(self.env.cv_action_onehot[action_sample[v][i]])
						share_drs.append(reward)
				
		return batch_ss, batch_ts, batch_as, batch_drs, share_ss, share_ts, share_as, share_drs, sample
	


	def _make_batch(self, sess, epoch):
		current_policy = self._prepare_current_policy(sess, epoch)

		# states = [
		#task1		[[---episode_1---],...,[---episode_n---]],
		#task2		[[---episode_1---],...,[---episode_n---]]
		#			]
		states, tasks, actions, rewards = self.rollout.rollout_batch(sess, self.PGNetwork, current_policy, epoch)     

		discounted_rewards = [[],[]]
		for index, task_rewards in enumerate(rewards):
			for ep_reward in task_rewards:
				discounted_rewards[index]+=self._discount_rewards(ep_reward)

		states[0] = np.concatenate(states[0])       
		states[1] = np.concatenate(states[1])
		tasks[0] = np.concatenate(tasks[0])     
		tasks[1] = np.concatenate(tasks[1])
		actions[0] = np.concatenate(actions[0])     
		actions[1] = np.concatenate(actions[1])
		rewards[0] = np.concatenate(rewards[0])
		rewards[1] = np.concatenate(rewards[1])
		
	
		batch_ss, batch_ts, batch_as, batch_drs, share_ss, share_ts, share_as, share_drs , samples= self._process_experience_normailize(sess, states, tasks, actions, discounted_rewards, current_policy) 

		return share_ss, share_ts, share_as, share_drs, batch_ss, batch_ts, batch_as, batch_drs, np.concatenate(rewards)
		
		
	def train(self, sess, saver):
		#run with saved initial model
		epoch = 1
		num_sample = 0
		while epoch < self.num_epochs + 1:
			print epoch
			
			#---------------------------------------------------------------------------------------------------------------------#	
			#ROLOUT SAMPLE
			share_ss, share_ts, share_as, share_drs, states_mb, tasks_mb, actions_mb, discounted_rewards_mb, rewards_mb  = self._make_batch(sess, epoch)
			#---------------------------------------------------------------------------------------------------------------------#	
			

			#---------------------------------------------------------------------------------------------------------------------#	
			#UPDATE NETWORK
			
			#base_line
			gradients = sess.run([self.PGNetwork.gvs], feed_dict={
																self.PGNetwork.inputs: states_mb,
																self.PGNetwork.inputt: tasks_mb,
																self.PGNetwork.actions: actions_mb,
																self.PGNetwork.rewards: discounted_rewards_mb 
																})
			if self.share_exp:
				if self.combine_gradent: #combine gradient
					gradients_share = sess.run([self.PGNetwork.gvs], feed_dict={
															self.PGNetwork.inputs: share_ss,
															self.PGNetwork.inputt: share_ts,
															self.PGNetwork.actions: share_as,
															self.PGNetwork.rewards: share_drs 
															})	
					#final_grad = weight*grad+(1-weight)*grad_share 
					for i, grad in enumerate(gradients[0]):
						for g,v  in enumerate(grad[0]):
							gradients[0][i][0][g]=self.share_exp_weight*gradients[0][i][0][g]+(1-self.share_exp_weight)*gradients_share[0][i][0][g]
				else: #combine sample
					gradients = sess.run([self.PGNetwork.gvs], feed_dict={
																self.PGNetwork.inputs: share_ss+states_mb,
																self.PGNetwork.inputt: share_ts+tasks_mb,
																self.PGNetwork.actions: share_as+actions_mb,
																self.PGNetwork.rewards: share_drs+discounted_rewards_mb 
																})	
				
			
			#update network weight from computed gradient	
			feed_dict = {}
			for i, grad in enumerate(gradients[0]):
			    feed_dict[self.PGNetwork.placeholder_gradients[i][0]] = grad[0]
			_ = sess.run([self.PGNetwork.train_opt],feed_dict=feed_dict)
			#---------------------------------------------------------------------------------------------------------------------#	
			


			#---------------------------------------------------------------------------------------------------------------------#	
			# WRITE TF SUMMARIES
			num_sample+=len(rewards_mb)

			total_reward_of_that_batch = np.sum(rewards_mb)
			mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, self.num_episide)
			summary = sess.run(self.write_op, feed_dict={
												self.PGNetwork.mean_reward: mean_reward_of_that_batch
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