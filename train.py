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
TIMER = str(datetime.now()).replace(' ', '_')

def training(test_time, map_index, num_task, share_exp, combine_gradent, num_episode):
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

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	log_folder = 'logs/' + TIMER

	if not os.path.isdir(log_folder):
		os.mkdir(log_folder)
		os.mkdir(os.path.join(log_folder, 'Combine_gradients'))
		os.mkdir(os.path.join(log_folder, 'Share_samples'))
		os.mkdir(os.path.join(log_folder, 'Non'))

	if share_exp:
		if combine_gradent: 
			writer = tf.summary.FileWriter(os.path.join(log_folder, 'Combine_gradients'))
		else:	
			writer = tf.summary.FileWriter(os.path.join(log_folder, 'Share_samples'))
	else:
		writer = tf.summary.FileWriter(os.path.join(log_folder, 'Non'))
	
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
										num_epochs 			= 2000,
										gamma 				= 0.99,
										plot_model 			= 100,
										save_model 			= 1,
										save_name 			= network_name_scope + '_' + test_name,
										num_episode 		= num_episode,
										share_exp 			= share_exp,
										combine_gradent 	= combine_gradent,
										share_exp_weight 	= 0.5,
										timer 				= TIMER
									)

	multitask_agent.train(sess, saver)
	sess.close()


if __name__ == '__main__':

	for i in range(1):
		training(test_time = i, map_index = 4, num_task = 2, share_exp = False, combine_gradent = False, num_episode = 20)
		training(test_time = i, map_index = 4, num_task = 2, share_exp = True, combine_gradent = True, num_episode = 20)
		training(test_time = i, map_index = 4, num_task = 2, share_exp = True, combine_gradent = False, num_episode = 20)