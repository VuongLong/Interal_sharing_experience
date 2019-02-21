# -*- coding: utf-8 -*-

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import os
import time

from utils import LinearSchedule
from rollout import Rollout
from env.constants import ACTION_SIZE
from env.scene_loader import PyGameDumpEnv

class MultitaskPolicy(object):

	def __init__(
			self,
			scene_name,
			networks,
			oracle_network,
			writer,
			write_op,
			gamma,
			plot_model,
			save_model,
			num_task,
			num_epochs,
			num_episodes,
			num_steps,
			share_exp,
			oracle
			):

		self.scene_name = scene_name
		self.networks = networks
		self.ZNetwork = oracle_network

		self.writer = writer
		self.write_op = write_op
		self.num_epochs = num_epochs
		self.gamma = gamma
		self.plot_model = plot_model
		self.save_model = save_model

		self.num_task = num_task
		self.num_episodes =  num_episodes
		self.num_steps = num_steps
		self.oracle = oracle
		self.share_exp = share_exp

		self.pretrain = [0] * self.num_task

		self.env = PyGameDumpEnv({"scene_name":scene_name,
								"task": 0,
								"success_reward": 1.0})

		self.rollout = Rollout(self.num_task, self.num_episodes, self.num_steps, self.scene_name)

	def _discount_rewards(self, episode_rewards, episode_states, episode_nexts, task, current_value):
		discounted_episode_rewards = np.zeros_like(episode_rewards)
		next_value = 0.0
		if episode_rewards[-1] == self.env.success_reward:
			next_value = 0.0
		else:
			next_value = current_value[episode_nexts[-1], task]

		for i in reversed(range(len(episode_rewards))):
			next_value = episode_rewards[i] + self.gamma * next_value  
			discounted_episode_rewards[i] = next_value

		return discounted_episode_rewards.tolist()

	def _GAE(self, episode_rewards, episode_states, episode_nexts, task, current_value):
		ep_GAE = np.zeros_like(episode_rewards)
		TD_error = np.zeros_like(episode_rewards)
		lamda=0.96

		next_value = 0.0
		if episode_rewards[-1] == self.env.success_reward:
			next_value = 0.0
		else:
			next_value = current_value[episode_nexts[-1], task]

		for i in reversed(range(len(episode_rewards))):
			TD_error[i] = episode_rewards[i]+self.gamma*next_value-current_value[episode_states[i], task]
			next_value = current_value[episode_states[i], task]

		ep_GAE[len(episode_rewards)-1] = TD_error[len(episode_rewards)-1]
		weight = self.gamma*lamda
		for i in reversed(range(len(episode_rewards)-1)):
			ep_GAE[i] += TD_error[i]+weight*ep_GAE[i+1]

		return ep_GAE.tolist()	


	def _prepare_current_policy(self, sess, epoch):
		current_policy = {}
		current_value = {}
		current_oracle = {}

		
		for loc in range(self.env.n_locations):
			for task in range(self.num_task):
				p, v = sess.run(
							[self.networks[task].actor.pi, self.networks[task].critic.value],
							feed_dict={
								self.networks[task].actor.inputs: [self.env.state(loc)],
								self.networks[task].critic.inputs: [self.env.state(loc)]})
			
				current_policy[loc, task] = p.ravel().tolist()
				current_value[loc, task] = v[0][0]
				current_oracle[loc, task, task] = [0.0, 1.0]

			if self.share_exp and not self.oracle:	
				for i in range (self.num_task-1):
					for j in range(i+1,self.num_task):	
						p = sess.run(
									self.ZNetwork[i,j].oracle, 
									feed_dict={
										self.ZNetwork[i,j].inputs: [self.env.state(loc)]})
						

						current_oracle[x,y,i,j] = p.ravel().tolist()
						boundary = 0.3
						if current_oracle[x,y,i,j][1]>boundary:
							current_oracle[x,y,i,j][1]-= boundary
							current_oracle[x,y,i,j][0]+= boundary
						else:
							current_oracle[x,y,i,j]=[1.0,0.0] 	
						current_oracle[x,y,j,i] = current_oracle[x,y,i,j]

		# if epoch%self.plot_model==0:
			# self.plot_figure.plot(current_policy, current_oracle, epoch)
						
		return current_policy, current_value, current_oracle

	

	def process_experience_normalize(self, epoch, states, actions, drewards, GAEs, next_states, current_policy, current_value, current_oracle):

		batch_ss, batch_as, batch_Qs, batch_Ts = [], [], [], []
		share_ss, share_as, share_Ts = [], [], []
		for task in range(self.num_task):
			batch_ss.append([])
			batch_as.append([])
			batch_Qs.append([])
			batch_Ts.append([])
			share_ss.append([])
			share_as.append([])
			share_Ts.append([])
		state_dict = {}
		count_dict = {}
		next_dict = {}

		use_gae = 1
		if self.share_exp:
			for task in range(self.num_task):
				for i, (state, action, actual_value, gae, next_state) in enumerate(zip(states[task], actions[task], drewards[task], GAEs[task], next_states[task])):	
					state_index = state
					advantage = actual_value - current_value[state, task]
					if use_gae==1:
						advantage = gae		
					#############################################################################################################
					if state_dict.get(state_index,-1)==-1:
						state_dict[state_index]=[]
						count_dict[state_index]=[]
						for tidx in range(self.num_task):
							state_dict[state_index].append([0.0] * self.env.action_size)
							count_dict[state_index].append([0] * self.env.action_size)
							
					state_dict[state_index][task][action]+=advantage
					count_dict[state_index][task][action]+=1
					next_dict[state_index, action]=[state, next_state]
					#############################################################################################################

		
		share_dict = {}
		mean_sa_dict = {}
		for task in range(self.num_task):
			if  self.pretrain[task]==0:
				for i, (state, action, actual_value, gae, next_state) in enumerate(zip(states[task], actions[task], drewards[task], GAEs[task], next_states[task])):	
					state_index = state

					count_a = 1
					advantage = actual_value - current_value[state, task]
					if use_gae==1:
						advantage = gae	
						
					#############################################################################################################
					if self.share_exp:
						mean_policy_action_task = []
						share_action = []
						
						if self.oracle:
							#GET SHARE INFORMATION FROM ORACLE MAP #############################################################
							if share_dict.get(state_index,-1)==-1:
								share_dict[state_index] = []
								share_info = bin(self.env.MAP[state[1]][state[0]])[2:].zfill(self.num_task*self.num_task)
								for tidx in range(self.num_task):
										share_dict[state_index].append([])
										share_info_task = share_info[tidx*self.num_task:(tidx+1)*self.num_task]
										for otidx in range(self.num_task):
											if share_info_task[otidx]=='1':
												share_dict[state_index][tidx].append(1)
											else:
												share_dict[state_index][tidx].append(0)
							####################################################################################################
							
							#TRANSFER ADVANTAGE OF TRAINED TASKS ###############################################################
							for tidx in range(self.num_task):
								if  self.pretrain[tidx]==1:
									if share_dict[state_index][task][tidx]==1:
										count_a += 1
										advantage += current_value[next_state[0],next_state[1],tidx] +(-0.01)- current_value[state[0],state[1],tidx]
							####################################################################################################

							#Calculate distrubtion of combination sample #######################################################	
							if mean_sa_dict.get((state_index, action),-1)==-1:
								mean_sa_dict[state_index, action] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

								for tidx in range(self.num_task):
									if  self.pretrain[tidx]==0:
										mean_policy_action = 0.0
										count = 0.0
										for otidx in range(self.num_task):
											if  self.pretrain[otidx]==0:
												if share_dict[state_index][tidx][otidx]==1:
													mean_policy_action+=(current_policy[state[0], state[1], otidx][action])
													count+=1	
										mean_policy_action/=count
										mean_sa_dict[state_index, action][tidx]=mean_policy_action
							####################################################################################################

						else:
							#TRANSFER ADVANTAGE OF TRAINED TASKS ###############################################################
							
							for tidx in range(self.num_task):
								if  self.pretrain[tidx]==1:
									share_action =np.random.choice(range(len(current_oracle[state[0],state[1],task,tidx])), 
													  p=np.array(current_oracle[state[0],state[1],task,tidx])/sum(current_oracle[state[0],state[1],task,tidx]))
									if share_action==1:				
										count_a += 1
										advantage += current_value[next_state[0],next_state[1],tidx] +(-0.01)- current_value[state[0],state[1],tidx]
							####################################################################################################

							#Calculate distrubtion of combination sample #######################################################	
							if mean_sa_dict.get((state_index, action),-1)==-1:
								mean_sa_dict[state_index, action] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
								
								for tidx in range(self.num_task):
									if  self.pretrain[tidx]==0:
										mean_policy_action = 0.0
										count = 0.0
										for otidx in range(self.num_task):
											if  self.pretrain[otidx]==0:
												mean_policy_action+=(current_oracle[state[0],state[1],otidx,tidx][1]*current_policy[state[0], state[1], otidx][action])
												count+=current_oracle[state[0],state[1],otidx,tidx][1]	
												
										mean_policy_action/=count
										mean_sa_dict[state_index, action][tidx]=mean_policy_action			
							####################################################################################################
							
						advantage /=count_a	
						for tidx in range(self.num_task):
							if  self.pretrain[tidx]==0:
								share_action = 1

								if self.oracle:
									share_action = share_dict[state_index][task][tidx]
								else:
									share_action = np.random.choice(range(len(current_oracle[state[0],state[1],task,tidx])), 
													  p=np.array(current_oracle[state[0],state[1],task,tidx])/sum(current_oracle[state[0],state[1],task,tidx]))
									
									if tidx > task:
										self.env.zmap[task, tidx][state[1]][state[0]] += share_action

								if share_action==1:				
									important_weight = current_policy[state[0],state[1],tidx][action]/mean_sa_dict[state_index, action][tidx]
									
									clip_important_weight = important_weight
									if clip_important_weight > 1.2:
										clip_important_weight = 1.2
									if clip_important_weight < 0.8:
										clip_important_weight = 0.8	

									if tidx==task:
										batch_ss[tidx].append(self.env.state(state_index))
										batch_as[tidx].append(self.env.cv_action_onehot[action])
										batch_Qs[tidx].append(actual_value)
										if (important_weight<=1.2 and important_weight>=0.8) or (clip_important_weight*advantage>important_weight*advantage):
											batch_Ts[tidx].append(important_weight*advantage)
										else:
											batch_Ts[tidx].append(advantage)

									else:
										if (important_weight<=1.2 and important_weight>=0.8) or (clip_important_weight*advantage>important_weight*advantage):

											share_ss[tidx].append(self.env.state(state_index))
											share_as[tidx].append(self.env.cv_action_onehot[action])
											share_Ts[tidx].append(important_weight*advantage)
					else:
						batch_ss[task].append(self.env.state(state_index))
						batch_as[task].append(self.env.cv_action_onehot[action])
						batch_Qs[task].append(actual_value)
						batch_Ts[task].append(advantage)

				#############################################################################################################
		z_ss, z_as, z_rs = {},{},{}
		if self.share_exp:
			for i in range (self.num_task-1):
				for j in range(i+1,self.num_task):
					z_ss[i,j] = []
					z_as[i,j] = []
					z_rs[i,j] = []
			for v in state_dict.keys():

				advantage_task = []	
				for i in range (self.num_task):
					advantage_task.append([])
					for action in range(self.env.action_size):	
						if  self.pretrain[i]==1:
							if next_dict.get((v, action),-1)==-1:
								advantage_task[i].append(0)
							else:
								s, n_s = next_dict[v, action]
								advantage_task_i_action = current_value[n_s[0],n_s[1],i]+(-0.01)-current_value[s[0],s[1],i]
								advantage_task[i].append(advantage_task_i_action)
						else:
							advantage_task[i].append(state_dict[v][i][action])
								
				for i in range (self.num_task-1):
					for j in range(i+1,self.num_task):
						for action in range(self.env.action_size):
							
							z_reward = 0.0
							if advantage_task[i][action]*advantage_task[j][action]>0:
								z_reward = min(abs(advantage_task[i][action]),abs(advantage_task[j][action]))
								z_action = [0,1]
							
							if advantage_task[i][action]*advantage_task[j][action]<0:
								z_reward = min(abs(advantage_task[i][action]),abs(advantage_task[j][action]))
								z_action = [1,0]

							if 	sum(count_dict[v][i])==0 and sum(count_dict[v][j])>0:
								z_reward = 0.001
								z_action = [1,0]
							
							if 	sum(count_dict[v][j])==0 and sum(count_dict[v][i])>0:
								z_reward = 0.001
								z_action = [1,0]
							
							if z_reward>0.0:
								z_ss[i,j].append(self.env.state(v))
								z_as[i,j].append(z_action)
								z_rs[i,j].append(z_reward)
	
		return batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts, z_ss, z_as, z_rs
	

	def _make_batch(self, sess, epoch, epsilon):
		current_policy, current_value, current_oracle = self._prepare_current_policy(sess, epoch)

		# states = [
		#task1		[[---episode_1---],...,[---episode_n---]],
		#task2		[[---episode_1---],...,[---episode_n---]]
		#			]
		states, tasks, actions, rewards, next_states, redundant_steps = self.rollout.rollout_batch(current_policy, epoch, epsilon)     

		discounted_rewards, GAEs = [], []
		for task in range(self.num_task):
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
			redundant_steps[task] = np.mean(redundant_steps[task])
		
		
		batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts, z_ss, z_as, z_rs = self.process_experience_normalize(epoch, states, actions, discounted_rewards, GAEs, next_states, current_policy, current_value, current_oracle) 

		return batch_ss, batch_as, batch_Qs, batch_Ts, share_ss, share_as, share_Ts, rewards, redundant_steps, z_ss, z_as, z_rs
		
		
	def train(self, sess, pretrain_dir):
		#run with saved initial model
		# for task_index in range(self.num_task):
		# 	if  self.pretrain[task_index]==1:
		# 		self.VNetwork[task_index].restore_model(sess,pretrain_dir+"pretrainv")
		# 		self.PGNetwork[task_index].restore_model(sess,pretrain_dir+"pretrainp")

		epoch = 1
		num_sample = 0
		start = time.time()

		epsilon_schedule = LinearSchedule(self.num_epochs, final_p=0.02)
		while epoch < self.num_epochs + 1:
			print("Elapsed time: {:.3f}: {}/{}".format((time.time() - start) / 3600, epoch, self.num_epochs), end='\r', flush=True)
			
			#---------------------------------------------------------------------------------------------------------------------#	
			#ROLOUT state_dict
			batch_ss, batch_as, batch_Qs, batch_Ts,\
			share_ss, share_as, share_Ts, rewards_mb,\
			redundant_steps, z_ss, z_as, z_rs  = self._make_batch(sess, epoch, epsilon_schedule.value(epoch + 1))
			#---------------------------------------------------------------------------------------------------------------------#	
		
			#---------------------------------------------------------------------------------------------------------------------#	
			#UPDATE NETWORK
			
			if self.share_exp:
				
				for i in range (self.num_task-1):
					for j in range(i+1,self.num_task):
						if len(z_ss[i,j])>0:
							sess.run([self.ZNetwork[i,j].train_opt], feed_dict={
																self.ZNetwork[i,j].inputs: z_ss[i,j],
																self.ZNetwork[i,j].actions: z_as[i,j], 
																self.ZNetwork[i,j].rewards: z_rs[i,j] 
																})	
				for task_index in range(self.num_task):
					if  self.pretrain[task_index]==0:
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
						# if epoch==self.num_epochs and self.oracle:
							#self.VNetwork[task_index].save_model(sess,pretrain_dir+"pretrainv")
							#self.PGNetwork[task_index].save_model(sess,pretrain_dir+"pretrainp")

							# if not self.oracle:
								# self.env.save_zmap(pretrain_dir+"z_{}/".format(self.num_episodes))

				if epoch % self.save_model == 0 and not self.oracle:
					for i in range(self.num_task-1):
						for j in range(i+1,self.num_task):	
							self.ZNetwork[i,j].save_model(sess, pretrain_dir+"z_{}/{}/".format(self.num_episodes, epoch))

			else:# base_line vanilla policy gradient
				for task_index in range(self.num_task):
					if  self.pretrain[task_index]==0:
						policy_loss, value_loss, _, _ = self.networks[task_index].learn(sess, 
																	actor_states = batch_ss[task_index],
																	advantages = batch_Ts[task_index],
																	actions = batch_as[task_index],
																	critic_states = batch_ss[task_index],
																	returns = batch_Qs[task_index]
																)

			#---------------------------------------------------------------------------------------------------------------------#	
			#---------------------------------------------------------------------------------------------------------------------#	
			# WRITE TF SUMMARIES
			sum_dict = {}

			total_reward_of_that_batch = 0.0
			task_redundant = []
			for task_index in range(self.num_task):
				if self.pretrain[task_index] == 0:
					num_sample += len(rewards_mb[task_index])
					total_reward_of_that_batch += np.sum(rewards_mb[task_index])
					task_redundant.append(redundant_steps[task_index])

				sum_dict[self.networks[task_index].mean_reward] = np.divide(np.sum(rewards_mb[task_index]), self.num_episodes)
				sum_dict[self.networks[task_index].mean_redundant] = redundant_steps[task_index]

			total_reward_of_that_batch /= self.pretrain.count(0)		
			mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, self.num_episodes)
			
			sum_dict[self.networks[0].total_mean_reward] = mean_reward_of_that_batch
			sum_dict[self.networks[0].average_mean_redundant] = np.mean(task_redundant)
			
			summary = sess.run(self.write_op, feed_dict=sum_dict)

			self.writer.add_summary(summary, num_sample)
			#self.writer.add_summary(summary, epoch)
			self.writer.flush()
			#---------------------------------------------------------------------------------------------------------------------#	


				
		
			#---------------------------------------------------------------------------------------------------------------------#		
			epoch += 1  