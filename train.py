import tensorflow as tf
import os 

from datetime import datetime
from network import PGNetwork, PGNetwork_deeper, PGNetwork_wider
from network import ZNetwork
from multitask_policy import MultitaskPolicy
from env.terrain import Terrain

NON = 1
SHARE = 2
ACTION_SIZE = 8

ask = input('Would you like to create new log folder?Y/n\n')
if ask == '' or ask.lower() == 'y':
	log_name = input("New folder's name: ")
	TIMER = str(log_name).replace(" ", "_")
	# TIMER = str(datetime.now()).replace(' ', '_')
else:
	TIMER = sorted(os.listdir('logs/'))[-1]

def training(test_time, map_index, num_task, share_exp, combine_gradent, sort_init, num_episode, learning_rate = 0.001, use_laser = False, change_arch = False):
	tf.reset_default_graph()
	
	if share_exp:
		if combine_gradent:
			network_name_scope = 'Combine_gradients'
		else:
			network_name_scope = 'Share_samples'
	else:
		network_name_scope = 'Non'

	env = Terrain(map_index, use_laser)
	policies = []
	
	if not change_arch:
		for i in range(num_task):
			policy_i = PGNetwork(
							state_size 		= env.cv_state_onehot.shape[1], 
							action_size 	= ACTION_SIZE, 
							learning_rate 	= learning_rate
							)
			policies.append(policy_i)
	else:
		policies.append(PGNetwork_wider(
							state_size 		= env.cv_state_onehot.shape[1], 
							action_size 	= ACTION_SIZE, 
							learning_rate 	= learning_rate
						))

		policies.append(PGNetwork_deeper(
								state_size 		= env.cv_state_onehot.shape[1], 
								action_size 	= ACTION_SIZE, 
								learning_rate 	= learning_rate
						))

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.2)

	sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	log_folder = 'logs/' + TIMER

	if not os.path.isdir(log_folder):
		os.mkdir(log_folder)
		os.mkdir(os.path.join(log_folder, 'Non_' + sort_init + "_laser_" + str(use_laser) + "_arch_" + str(change_arch)))
		os.mkdir(os.path.join(log_folder, 'Combine_gradients'))
		os.mkdir(os.path.join(log_folder, 'Share_samples'))
		
	if share_exp:
		if combine_gradent: 
			writer = tf.summary.FileWriter(os.path.join(log_folder, 'Combine_gradients'))
		else:	
			writer = tf.summary.FileWriter(os.path.join(log_folder, 'Share_samples'))
	else:
		writer = tf.summary.FileWriter(os.path.join(log_folder, 'Non_' + sort_init + "_laser_" + str(use_laser) + "_arch_" + str(change_arch)))
	
	test_name =  "map_" + str(map_index) + "_test_" + str(test_time)
	tf.summary.scalar(test_name, tf.reduce_mean([policy.mean_reward for policy in policies], 0))
	write_op = tf.summary.merge_all()

	multitask_agent = MultitaskPolicy(
										map_index 			= map_index,
										policies 			= policies,
										writer 				= writer,
										write_op 			= write_op,
										action_size 		= 8,
										num_task 			= num_task,
										num_iters 			= 1500000,
										gamma 				= 0.99,
										plot_model 			= 50,
										save_model 			= 100,
										save_name 			= network_name_scope + '_' + test_name + '_' + sort_init + "_laser_" + str(use_laser) + "_arch_" + str(change_arch),
										num_episode 		= num_episode,
										share_exp 			= share_exp,
										combine_gradent 	= combine_gradent,
										share_exp_weight 	= 0.5,
										sort_init			= sort_init,
										use_laser			= use_laser,
										timer 				= TIMER
									)

	multitask_agent.train(sess, saver)
	sess.close()


if __name__ == '__main__':

	# for i in range(5):
	# 	training(test_time = i, map_index = 4, num_task = 2, share_exp = False, combine_gradent = False, sort_init = 'local', num_episode = 20)
	# 	training(test_time = i, map_index = 4, num_task = 2, share_exp = False, combine_gradent = False, sort_init = 'global', num_episode = 20)
	# 	training(test_time = i, map_index = 4, num_task = 2, share_exp = False, combine_gradent = False, sort_init = 'none', num_episode = 20)

	for i in range(5):
		# training(test_time = i, map_index = 6, num_task = 2, share_exp = False, combine_gradent = False, sort_init = 'none', num_episode = 20, learning_rate = 0.0001, use_laser = True)
		# training(test_time = i, map_index = 6, num_task = 2, share_exp = False, combine_gradent = False, sort_init = 'none', num_episode = 20, use_laser = False, change_arch = True)
		training(test_time = i, map_index = 5, num_task = 2, share_exp = False, combine_gradent = False, sort_init = 'none', num_episode = 20, use_laser = False)
		# training(test_time = i, map_index = 6, num_task = 2, share_exp = False, combine_gradent = False, sort_init = 'global', num_episode = 20, use_laser = False)
		# training(test_time = i, map_index = 6, num_task = 2, share_exp = False, combine_gradent = True, sort_init = 'none', num_episode = 10, use_laser = False)
		training(test_time = i, map_index = 5, num_task = 2, share_exp = True, combine_gradent = False, sort_init = 'none', num_episode = 20, use_laser = False)
		# training(test_time = i, map_index = 6, num_task = 2, share_exp = False, combine_gradent = True, sort_init = 'none', num_episode = 20, use_laser = False)

		# training(test_time = i, map_index = 4, num_task = 2, share_exp = True, combine_gradent = True, num_episode = 20)
		# training(test_time = i, map_index = 4, num_task = 2, share_exp = True, combine_gradent = False, num_episode = 20)