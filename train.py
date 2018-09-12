import tensorflow as tf
import os 

from datetime import datetime
from network import PGNetwork
from network import ZNetwork
from multitask_policy import MultitaskPolicy
from env.terrain import Terrain

NON = 1
SHARE = 2
ACTION_SIZE = 8

ask = input('Would you like to create new log folder?Y/n')
if ask == '' or ask.lower() == 'y':
	TIMER = str(datetime.now()).replace(' ', '_')
else:
	TIMER = sorted(os.listdir('logs/'))[-1]

def training(test_time, map_index, num_task, share_exp, combine_gradent, sort_init, num_episode):
	tf.reset_default_graph()
	
	learning_rate = 0.001

	if share_exp:
		if combine_gradent:
			network_name_scope = 'Combine_gradients'
		else:
			network_name_scope = 'Share_samples'
	else:
		network_name_scope = 'Non'

	env = Terrain(map_index)
	policies = []
	for i in range(num_task):
		policy_i = PGNetwork(
						state_size 		= env.cv_state_onehot.shape[0], 
						action_size 	= ACTION_SIZE, 
						learning_rate 	= learning_rate
						)
		policies.append(policy_i)

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	log_folder = 'logs/' + TIMER

	if not os.path.isdir(log_folder):
		os.mkdir(log_folder)

	if share_exp:
		if combine_gradent: 
			os.mkdir(os.path.join(log_folder, 'Combine_gradients'))
			writer = tf.summary.FileWriter(os.path.join(log_folder, 'Combine_gradients'))
		else:	
			os.mkdir(os.path.join(log_folder, 'Share_samples'))
			writer = tf.summary.FileWriter(os.path.join(log_folder, 'Share_samples'))
	else:
		os.mkdir(os.path.join(log_folder, 'Non_' + sort_init))
		writer = tf.summary.FileWriter(os.path.join(log_folder, 'Non_' + sort_init))
	
	test_name =  "map_" + str(map_index) + "_test_" + str(test_time) + "_sortinit_" + sort_init
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
										plot_model 			= 100,
										save_model 			= 100,
										save_name 			= network_name_scope + '_' + test_name,
										num_episode 		= num_episode,
										share_exp 			= share_exp,
										combine_gradent 	= combine_gradent,
										share_exp_weight 	= 0.5,
										sort_init			= sort_init,
										timer 				= TIMER
									)

	multitask_agent.train(sess, saver)
	sess.close()


if __name__ == '__main__':

	for i in range(5):
		training(test_time = i, map_index = 4, num_task = 2, share_exp = False, combine_gradent = False, sort_init = 'local', num_episode = 20)
		training(test_time = i, map_index = 4, num_task = 2, share_exp = False, combine_gradent = False, sort_init = 'global', num_episode = 20)
		training(test_time = i, map_index = 4, num_task = 2, share_exp = False, combine_gradent = False, sort_init = 'none', num_episode = 20)

		# training(test_time = i, map_index = 4, num_task = 2, share_exp = True, combine_gradent = True, num_episode = 20)
		# training(test_time = i, map_index = 4, num_task = 2, share_exp = True, combine_gradent = False, num_episode = 20)