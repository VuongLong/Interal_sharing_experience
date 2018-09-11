import tensorflow as tf
import os 

from datetime import datetime
from network import PGNetwork
from network import ZNetwork
from multitask_policy import MultitaskPolicy

NON = 1
SHARE = 2
log_folder = 'logs_' + str(datetime.now()).replace(' ', '_')

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

	policy = PGNetwork(
					state_size 		= 323, 
					task_size 		= 2, 
					action_size 	= 8, 
					learning_rate 	= learning_rate
					)

	oracle = ZNetwork(
					state_size 		= 323, 
					action_size 	= 2, 
					learning_rate 	= learning_rate
					)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

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

	# if num_task == 1:
	# 	writer = tf.summary.FileWriter("logs/1_task")
	# else:
	# 	writer = tf.summary.FileWriter("logs/2_tasks")
	
	test_name =  "map_" + str(map_index) + "_test_" + str(test_time)		
	tf.summary.scalar(test_name, policy.mean_reward)
	write_op = tf.summary.merge_all()

	multitask_agent = MultitaskPolicy(
										map_index 			= map_index,
										policy 				= policy,
										oracle 				= oracle,
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
									)

	# Continue train
	#saver.restore(sess, "./models/model.ckpt")
	
	# Retrain with saving initial model 
	#saver.save(sess, "./model_init_onehot/model.ckpt")
	# saver.restore(sess, "./model_init_onehot/model.ckpt")
	
	multitask_agent.train(sess, saver)
	sess.close()


if __name__ == '__main__':

	# for i in range(10):
		# training(test_time = i, map_index = 5, num_task = 1, share_exp = False, combine_gradent = False, num_episode = 20)
		# training(test_time = i, map_index = 5, num_task = 2, share_exp = False, combine_gradent = False, num_episode = 20)

	for i in range(5):
		training(test_time = i, map_index = 4, num_task = 2, share_exp = False, combine_gradent = False, num_episode = 20)
		training(test_time = i, map_index = 4, num_task = 2, share_exp = True, combine_gradent = True, num_episode = 20)
		training(test_time = i, map_index = 4, num_task = 2, share_exp = True, combine_gradent = False, num_episode = 20)
	

'''
NOTE:
	file Multitask.py: line 144-158
	Future: 
		should task important weight or not
		should keep sample in non-sharing areas to avoid bias or not
	=>  read related work and document

	Test:
		tunning 3 hyperparameters: 
			num_episode
			share_exp
			combine_gradent
'''
