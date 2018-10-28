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
			oracle_network,
			writer,
			write_op,
			num_epochs,			
			gamma,
			plot_model,
			save_model,
			save_name,

			num_episide,
			share_exp,
			oracle,
			share_exp_weight
			):

		self.map_index = map_index
		self.PGNetwork = policy
		self.VNetwork = value
		self.ZNetwork = oracle_network

		self.writer = writer
		self.write_op = write_op
		self.num_epochs = num_epochs
		self.gamma = gamma
		self.save_name = save_name
		self.plot_model = plot_model
		self.save_model = save_model

		self.num_episide =  num_episide
		self.oracle = oracle
		self.share_exp = share_exp
		self.share_exp_weight = share_exp_weight

		self.env = Terrain(self.map_index)

		self.plot_figure = PlotFigure(self.save_name, self.env)

		self.rollout = Rollout(number_episode = self.num_episide, map_index = self.map_index)
		self.epsilon = 0.2

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
		current_oracle = {}
		range_x = [0, self.env.bounds_x[1]+1]
		range_y = [0, self.env.bounds_y[1]+1]

		
		for x in range(range_x[0]+1,range_x[1]):
			for y in range(range_y[0]+1,range_y[1]):
				if self.env.MAP[y][x]!=0:
					for task in range(self.env.num_task):
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
						current_oracle[x,y,task,task] = [0.0,1.0]

					for i in range (self.env.num_task-1):
						for j in range(i+1,self.env.num_task):	
							p = sess.run(
										self.ZNetwork[i,j].oracle, 
										feed_dict={
											self.ZNetwork[i,j].inputs: [self.env.cv_state_onehot[x+(y-1)*self.env.bounds_x[1]-1].tolist()]})
							

							current_oracle[x,y,i,j] = p.ravel().tolist()
							boundary = 0.3
							if current_oracle[x,y,i,j][1]>boundary:
								current_oracle[x,y,i,j][1]-= boundary
								current_oracle[x,y,i,j][0]+= boundary
							else:
								current_oracle[x,y,i,j]=[1.0,0,0] 	
							current_oracle[x,y,j,i] = current_oracle[x,y,i,j]

		if epoch%self.plot_model==0:
			self.plot_figure.plot(current_policy, current_oracle, epoch)
						
		return current_policy, current_value, current_oracle

	

	def _process_experience_normailize(self, epoch, states, actions, drewards, GAEs, next_states, current_policy, current_value, current_oracle):

		batch_ss, batch_as, batch_Qs, batch_Ts = [], [], [], []
		share_ss, share_as, share_Ts = [], [], []
		for task in range(self.env.num_task):
			batch_ss.append([])
			batch_as.append([])
			batch_Qs.append([])
			batch_Ts.append([])
			share_ss.append([])
			share_as.append([])
			share_Ts.append([])
		state_dict = {}
		count_dict = {}


		for task in range(self.env.num_task):
			for i, (state, action, actual_value, gae, next_state) in enumerate(zip(states[task], actions[task], drewards[task],GAEs[task], next_states[task])):	
				state_index = state[0]+(state[1]-1)*self.env.bounds_x[1]-1

				#############################################################################################################
				if state_dict.get(state_index,-1)==-1:
					state_dict[state_index]=[]
					count_dict[state_index]=[]
					for tidx in range(self.env.num_task):
						state_dict[state_index].append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
						count_dict[state_index].append([0, 0, 0, 0, 0, 0, 0, 0])
								
				state_dict[state_index][task][action]+=gae
				count_dict[state_index][task][action]+=1
				#############################################################################################################

		
		share_dict = {}
		mean_sa_dict = {}
		for task in range(self.env.num_task):
			for i, (state, action, actual_value, gae, next_state) in enumerate(zip(states[task], actions[task], drewards[task],GAEs[task], next_states[task])):	
				state_index = state[0]+(state[1]-1)*self.env.bounds_x[1]-1

				#############################################################################################################
				if self.share_exp:
					mean_policy_action_task = []
					share_action = []
					
					if self.oracle:
						#GET SHARE INFORMATION FROM ORACLE MAP #############################################################
						if share_dict.get(state_index,-1)==-1:
							share_dict[state_index] = []
							share_info = bin(self.env.MAP[state[1]][state[0]])[2:].zfill(self.env.num_task*self.env.num_task)
							for tidx in range(self.env.num_task): 
								share_dict[state_index].append([])
								share_info_task = share_info[tidx*self.env.num_task:(tidx+1)*self.env.num_task]
								for otidx in range(self.env.num_task):
									if share_info_task[otidx]=='1':
										share_dict[state_index][tidx].append(1)
									else:
										share_dict[state_index][tidx].append(0)
						#####################################################################################################
						
						#Calculate distrubtion of combination sample ########################################################	
						if mean_sa_dict.get((state_index, action),-1)==-1:
							mean_sa_dict[state_index, action] = []

							for tidx in range(self.env.num_task):
								mean_policy_action = 0.0
								count = 0.0
								for otidx in range(self.env.num_task):
									if share_dict[state_index][tidx][otidx]==1:
										if otidx==tidx or count_dict[state_index][otidx][action]>0:
											mean_policy_action+=(current_policy[state[0], state[1], otidx][action])
											count+=1	
								mean_policy_action/=count
								mean_sa_dict[state_index, action].append(mean_policy_action)
						#####################################################################################################

					else:
						#GET SHARE INFORMATION FROM Z MAP #############################################################
						if share_dict.get(state_index,-1)==-1: 
							share_dict[state_index]=[]
							for tidx in range(self.env.num_task):
								share_dict[state_index].append([])
								for otidx in range(self.env.num_task):
								 	share_dict[state_index][tidx].append(0)
								share_dict[state_index][tidx][tidx]=1
										
							for tidx in range (self.env.num_task-1):
								for otidx in range(i+1,self.env.num_task):
									share_action =np.random.choice(range(len(current_oracle[state[0],state[1],otidx,tidx])), 
											  p=np.array(current_oracle[state[0],state[1],otidx,tidx])/sum(current_oracle[state[0],state[1],otidx,tidx]))
									share_dict[state_index][tidx][otidx] = share_action
									share_dict[state_index][otidx][tidx] = share_action
						#####################################################################################################
							
						#Calculate distrubtion of combination sample ########################################################	
						if mean_sa_dict.get((state_index, action),-1)==-1:
							mean_sa_dict[state_index, action] = []

							for tidx in range(self.env.num_task):
								mean_policy_action = 0.0
								count = 0.0
								for otidx in range(self.env.num_task):
									#if share_dict[state_index][tidx][otidx]==1:
									if otidx==tidx or count_dict[state_index][otidx][action]>0:
										mean_policy_action+=(current_oracle[state[0],state[1],otidx,tidx][1]*current_policy[state[0], state[1], otidx][action])
										count+=current_oracle[state[0],state[1],otidx,tidx][1]	
											
								mean_policy_action/=count
								mean_sa_dict[state_index, action].append(mean_policy_action)			
						#####################################################################################################
						

					for tidx in range(self.env.num_task):
						#if share_dict[state_index][task][tidx]==1:
						share_action =np.random.choice(range(len(current_oracle[state[0],state[1],task,tidx])), 
											  p=np.array(current_oracle[state[0],state[1],task,tidx])/sum(current_oracle[state[0],state[1],task,tidx]))
						if share_action==1:				
							important_weight = current_policy[state[0],state[1],tidx][action]/mean_sa_dict[state_index, action][tidx]
							
							clip_important_weight = important_weight
							if clip_important_weight > 1.2:
								clip_important_weight = 1.2
							if clip_important_weight < 0.8:
								clip_important_weight = 0.8	

							advantage = actual_value - current_value[state[0],state[1],task]
							advantage = gae
							if (important_weight<=1.2 and important_weight>=0.8) or (clip_important_weight*advantage>important_weight*advantage):
								if tidx==task:
									batch_ss[tidx].append(self.env.cv_state_onehot[state_index])
									batch_as[tidx].append(self.env.cv_action_onehot[action])
									batch_Qs[tidx].append(actual_value)
									batch_Ts[tidx].append(important_weight*advantage)

								else:
									share_ss[tidx].append(self.env.cv_state_onehot[state_index])
									share_as[tidx].append(self.env.cv_action_onehot[action])
									share_Ts[tidx].append(important_weight*advantage)
				else:

					advantage = actual_value - current_value[state[0],state[1],task]
					advantage=gae
					batch_ss[task].append(self.env.cv_state_onehot[state_index])
					batch_as[task].append(self.env.cv_action_onehot[action])
					batch_Qs[task].append(actual_value)
					batch_Ts[task].append(advantage)

				#############################################################################################################
		
		z_ss, z_as, z_rs = {},{},{}
		for i in range (self.env.num_task-1):
			for j in range(i+1,self.env.num_task):
				z_ss[i,j] = []
				z_as[i,j] = []
				z_rs[i,j] = []
		for v in state_dict.keys():
			
			for i in range (self.env.num_task):
				if count_dict[v][i][action]>0:
					state_dict[v][i][action] = state_dict[v][i][action]/count_dict[v][i][action]

			for i in range (self.env.num_task-1):
				for j in range(i+1,self.env.num_task):
					for action in range(self.env.action_size):
					
						z_reward = 0.0
						#if state_dict[v][0][action]>0 and state_dict[v][1][action]>0:
						if state_dict[v][i][action]*state_dict[v][j][action]>0:
							z_reward = min(abs(state_dict[v][i][action]),abs(state_dict[v][j][action]))
							z_action = [0,1]
						
						if state_dict[v][i][action]*state_dict[v][j][action]<0:
							z_reward = min(abs(state_dict[v][i][action]),abs(state_dict[v][j][action]))
							z_action = [1,0]

						if 	sum(count_dict[v][i])==0 and sum(count_dict[v][j])>0:
							z_reward = 0.001
							z_action = [1,0]
						
						if 	sum(count_dict[v][j])==0 and sum(count_dict[v][i])>0:
							z_reward = 0.001
							z_action = [1,0]
						
						if z_reward>0.0:
							z_ss[i,j].append(self.env.cv_state_onehot[v].tolist())
							z_as[i,j].append(z_action)
							z_rs[i,j].append(z_reward)

			#print state_dict[v][0]		
			#print state_dict[v][1]		
		# print len(z_ss)			
		return batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts, z_ss, z_as, z_rs
	

	def _make_batch(self, sess, epoch):
		current_policy, current_value, current_oracle = self._prepare_current_policy(sess, epoch)

		# states = [
		#task1		[[---episode_1---],...,[---episode_n---]],
		#task2		[[---episode_1---],...,[---episode_n---]]
		#			]
		states, tasks, actions, rewards, next_states= self.rollout.rollout_batch(sess, self.PGNetwork, current_policy, epoch)     

		discounted_rewards, GAEs = [], []
		for task in range(self.env.num_task):
			discounted_rewards.append([])
			GAEs.append([])
			for ep_state, ep_next, ep_reward in zip(states[task], next_states[task], rewards[task]):	
				discounted_rewards[task]+=self._discount_rewards(ep_reward, ep_state, ep_next, task, current_value)
				GAEs[task]+=self._GAE(ep_reward, ep_state, ep_next, task, current_value)
			
			states[task] = np.concatenate(states[task])       
			tasks[task] = np.concatenate(tasks[task])     
			actions[task] = np.concatenate(actions[task])     
			rewards[task] = np.concatenate(rewards[task])
			next_states[task] = np.concatenate(next_states[task])
		
		
		batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts, z_ss, z_as, z_rs = self._process_experience_normailize(epoch, states, actions, discounted_rewards, GAEs, next_states, current_policy, current_value, current_oracle) 

		return batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts, np.concatenate(rewards), z_ss, z_as, z_rs
		
		
	def train(self, sess, saver):
		#run with saved initial model
		epoch = 1
		num_sample = 0
		while epoch < self.num_epochs + 1:
			print epoch
			
			#---------------------------------------------------------------------------------------------------------------------#	
			#ROLOUT state_dict
			batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts, rewards_mb, z_ss, z_as, z_rs  = self._make_batch(sess, epoch)
			#---------------------------------------------------------------------------------------------------------------------#	
		
			#---------------------------------------------------------------------------------------------------------------------#	
			#UPDATE NETWORK
			
			if self.share_exp:
				
				for i in range (self.env.num_task-1):
					for j in range(i+1,self.env.num_task):
						if len(z_ss[i,j])>0:
							sess.run([self.ZNetwork[i,j].train_opt], feed_dict={
																self.ZNetwork[i,j].inputs: z_ss[i,j],
																self.ZNetwork[i,j].actions: z_as[i,j], 
																self.ZNetwork[i,j].rewards: z_rs[i,j] 
																})	
				for task_index in range(self.env.num_task):
					
					sess.run([self.VNetwork[task_index].train_opt], feed_dict={
																self.VNetwork[task_index].inputs: batch_ss[task_index],
																self.VNetwork[task_index].rewards: batch_Qs[task_index] 
																})	
					share_ss[task_index]+=batch_ss[task_index]
					share_as[task_index]+=batch_as[task_index]
					share_Ts[task_index]+=batch_Ts[task_index]

					sess.run([self.PGNetwork[task_index].train_opt], feed_dict={
																self.PGNetwork[task_index].inputs: share_ss[task_index],
																self.PGNetwork[task_index].actions: share_as[task_index],
																self.PGNetwork[task_index].rewards: share_Ts[task_index]
																})	
					

			else:
				for task_index in range(self.env.num_task):
					#base_line
					loss,_ =sess.run([self.VNetwork[task_index].loss, self.VNetwork[task_index].train_opt], feed_dict={
																self.VNetwork[task_index].inputs: batch_ss[task_index],
																self.VNetwork[task_index].rewards: batch_Qs[task_index] 
																})	
		
					sess.run([self.PGNetwork[task_index].train_opt], feed_dict={
																		self.PGNetwork[task_index].inputs: batch_ss[task_index],
																		self.PGNetwork[task_index].actions: batch_as[task_index],
																		self.PGNetwork[task_index].rewards: batch_Ts[task_index]
																		})
					
			#---------------------------------------------------------------------------------------------------------------------#	
			


			#---------------------------------------------------------------------------------------------------------------------#	
			# WRITE TF SUMMARIES
			num_sample+=len(rewards_mb) / self.env.num_task

			total_reward_of_that_batch = np.sum(rewards_mb) / self.env.num_task
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