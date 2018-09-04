# -*- coding: utf-8 -*-
import tensorflow as tf      # Deep Learning library

from network import PGNetwork
from network import ZNetwork
from multitask_policy import MultitaskPolicy

NON = 1
SHARE = 2

def training(test_time, map_index, share_exp, combine_gradent, num_episide):
	tf.reset_default_graph()
	
	learning_rate= 0.001

	if share_exp:
		if combine_gradent:
			network_name_scope = 'Combine_gradients'
		else:
			network_name_scope = 'Share_samples'
	else:
		network_name_scope = 'Non'

	policy = PGNetwork(
					state_size = 323, 
					task_size = 2, 
					action_size = 8, 
					learning_rate = learning_rate
					)

	oracle = ZNetwork(
					state_size = 323, 
					action_size = 2, 
					learning_rate = learning_rate
					)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	if share_exp:
		if combine_gradent: 
			writer = tf.summary.FileWriter("./plot/log/Combine_gradients")
		else:	
			writer = tf.summary.FileWriter("./plot/log/Share_samples")
	else:
		writer = tf.summary.FileWriter("./plot/log/Non")
	
	test_name =  "map_"+str(map_index)+"_test_"+str(test_time)		
	tf.summary.scalar(test_name, policy.mean_reward)
	write_op = tf.summary.merge_all()

	multitask_agent = MultitaskPolicy(
									map_index = map_index,
									policy = policy,
									oracle = oracle,
									writer = writer,
									write_op = write_op,
									state_size = 2,
									action_size = 8,
									task_size = 2,
									num_epochs = 1000,
									gamma = 0.99,
									plot_model = 50,
									save_model = 100,
									save_name = network_name_scope+'_'+test_name,
									num_episide = num_episide,
									share_exp = share_exp,
									combine_gradent = combine_gradent,
									share_exp_weight = 0.5,
									)

	# Continue train
	#saver.restore(sess, "./models/model.ckpt")
	
	# Retrain with saving initial model 
	#saver.save(sess, "./model_init_onehot/model.ckpt")
	saver.restore(sess, "./model_init_onehot/model.ckpt")
	
	multitask_agent.train(sess, saver)
	sess.close()


if __name__ == '__main__':
	for i in range(10):
		training(test_time=i, map_index=4, share_exp = False, combine_gradent=False, num_episide = 20)
		training(test_time=i, map_index=4, share_exp = True, combine_gradent=False, num_episide = 20)
		training(test_time=i, map_index=4, share_exp = True, combine_gradent=True, num_episide = 20)
	

'''
NOTE:
	file Multitask.py: line 144-158
	Future: 
		should task important weight or not
		should keep sample in non-sharing areas to avoid bias or not
	=> read related work and doccument

	Test:
		tunning 3 hyperparameters: 
			num_episide
			share_exp
			combine_gradent
'''