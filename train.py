# -*- coding: utf-8 -*-
import tensorflow as tf      # Deep Learning library

from network import PGNetwork
from network import ZNetwork
from constant import ACTION_SIZE
from multitask_policy import MultitaskPolicy
from env.terrain import Terrain

NON = 1
SHARE = 2

def training(test_time, map_index, share_exp, combine_gradent, num_episide):
	tf.reset_default_graph()
	
	learning_rate= 0.005

	if share_exp:
		if combine_gradent:
			network_name_scope = 'Combine_gradients'
		else:
			network_name_scope = 'Share_samples'
	else:
		network_name_scope = 'Non'
	env = Terrain(map_index)
	
	policy = []
	for i in range(2):	
		policy_i = PGNetwork(
						state_size = env.size_m, 
						task_size = 2, 
						action_size = ACTION_SIZE, 
						learning_rate = learning_rate,
						name = "PGNetwork_"+str(i)
						)
		policy.append(policy_i)

	oracle = ZNetwork(
					state_size = env.size_m, 
					action_size = 2, 
					learning_rate = learning_rate
					)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	if share_exp:
		if combine_gradent: 
			writer = tf.summary.FileWriter("../plot/log/Combine_gradients"+"_"+str(num_episide))
		else:	
			writer = tf.summary.FileWriter("../plot/log/Share_samples"+"_"+str(num_episide))
	else:
		writer = tf.summary.FileWriter("../plot/log/Non"+"_"+str(num_episide))
	
	test_name =  "map_"+str(map_index)		
	tf.summary.scalar(test_name, policy[0].mean_reward)
	write_op = tf.summary.merge_all()

	multitask_agent = MultitaskPolicy(
									map_index = map_index,
									policy = policy,
									oracle = oracle,
									writer = writer,
									write_op = write_op,
									num_epochs = 1000,
									gamma = 1.0,
									plot_model = 100,
									save_model = 10000,
									save_name = network_name_scope+'_'+test_name+'_'+str(num_episide),
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
	map_index = 4
	for i in range(6):
		num_ep = 4*(i+1)
		training(test_time=i, map_index=map_index, share_exp = False, combine_gradent=False, num_episide =num_ep)
		training(test_time=i, map_index=map_index, share_exp = True, combine_gradent=False, num_episide = num_ep)
		#training(test_time=i, map_index=map_index, share_exp = True, combine_gradent=True, num_episide = num_ep)
	

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