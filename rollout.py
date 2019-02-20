import numpy as np           # Handle matrices
import threading

from rollout_thread import RolloutThread

class Rollout(object):
	
	def __init__(
		self,
		num_task,
		num_episode,
		num_step,
		scene_name):
		
		self.scene_name = scene_name
		
		self.num_step = num_step
		self.num_episode = num_episode
		self.num_task = num_task

		self.states, self.tasks, self.actions, self.rewards, self.next_states, self.redundant_steps = [],[],[],[],[],[]

		for task in range(self.num_task):
			self.states.append([])
			self.tasks.append([])
			self.actions.append([])
			self.rewards.append([])
			self.next_states.append([])
			self.redundant_steps.append([])
			for i in range(self.num_episode):
				self.states[task].append([])
				self.tasks[task].append([])
				self.actions[task].append([])
				self.rewards[task].append([])
				self.next_states[task].append([])
				self.redundant_steps[task].append([])

	def _rollout_process(self, index, task, current_policy, epsilon):
		thread_rollout = RolloutThread(
									task = task,
									num_step = self.num_step,
									policy = current_policy,
									scene_name = self.scene_name)

		ep_states, ep_tasks, ep_actions, ep_rewards, ep_next_states, ep_redundant_steps = thread_rollout.rollout(epsilon)
		
		self.states[task][index]=ep_states
		self.tasks[task][index]=ep_tasks
		self.actions[task][index]=ep_actions
		self.rewards[task][index]=ep_rewards
		self.next_states[task][index]=ep_next_states
		self.redundant_steps[task][index] = ep_redundant_steps


	def rollout_batch(self, policy, epoch, epsilon):
		self.states, self.tasks, self.actions, self.rewards, self.next_states, self.redundant_steps = [],[],[],[],[],[]
		for task in range(self.num_task):
			self.states.append([])
			self.tasks.append([])
			self.actions.append([])
			self.rewards.append([])
			self.next_states.append([])
			self.redundant_steps.append([])
			for i in range(self.num_episode):
				self.states[task].append([])
				self.tasks[task].append([])
				self.actions[task].append([])
				self.rewards[task].append([])
				self.next_states[task].append([])
				self.redundant_steps[task].append([])

		train_threads = []
		for task in range(self.num_task):
			for index in range(self.num_episode):
				train_threads.append(threading.Thread(target=self._rollout_process, args=(index, task, policy, epsilon)))

		# start each training thread
		for t in train_threads:
			t.start()

		# wait for all threads to finish
		for t in train_threads:
			t.join()		

		return self.states, self.tasks, self.actions, self.rewards, self.next_states, self.redundant_steps