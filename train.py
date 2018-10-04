# -*- coding: utf-8 -*-
import tensorflow as tf      # Deep Learning library

from network import PGNetwork
from network import VNetwork
from network import ZNetwork

from constant import ACTION_SIZE
from multitask_policy import MultitaskPolicy
from env.terrain import Terrain

NON = 1
SHARE = 2

def training(test_time, map_index, share_exp, oracle, num_episide):
	tf.reset_default_graph()
	
	learning_rate= 0.005

	if share_exp:
		if oracle:
			network_name_scope = 'Combine_gradients'
		else:
			network_name_scope = 'Share_samples'
	else:
		network_name_scope = 'Non'
	env = Terrain(map_index)
	
	policy = []
	value = []
	for i in range(2):	
		policy_i = PGNetwork(
						state_size = env.size_m, 
						task_size = 2, 
						action_size = ACTION_SIZE, 
						learning_rate = learning_rate,
						name = "PGNetwork_"+str(i)
						)
		policy.append(policy_i)
		value_i = VNetwork(
						state_size = env.size_m, 
						task_size = 2, 
						action_size = ACTION_SIZE, 
						learning_rate = learning_rate,
						name = "VNetwork_"+str(i)
						)
		value.append(value_i)
		
	oracle_network = ZNetwork(
					state_size = env.size_m, 
					action_size = 2, 
					learning_rate = learning_rate,
					name = "oracle"
					)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	if share_exp:
		if oracle: 
			writer = tf.summary.FileWriter("../plot/log/oracle"+"_"+str(num_episide))
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
									value = value,
									oracle_network = oracle_network,
									writer = writer,
									write_op = write_op,
									num_epochs = 2000,
									gamma = 0.99,
									plot_model = 50,
									save_model = 10000,
									save_name = network_name_scope+'_'+test_name+'_'+str(num_episide),
									num_episide = num_episide,
									share_exp = share_exp,
									oracle = oracle,
									share_exp_weight = 0.5,
									)

	# Continue train
	#saver.restore(sess, "./models/model.ckpt")
	
	# Retrain with saving initial model 
	#saver.save(sess, "./model_init_onehot/model.ckpt")
	#saver.restore(sess, "./model_init_onehot/model.ckpt")
	
	multitask_agent.train(sess, saver)
	sess.close()


if __name__ == '__main__':
	map_index = 4
	for i in range(3):
		num_ep = 8*(i+1)
		training(test_time=i, map_index=map_index, share_exp = False, oracle=False, num_episide =num_ep)
		training(test_time=i, map_index=map_index, share_exp = True, oracle=False, num_episide = num_ep)
		training(test_time=i, map_index=map_index, share_exp = True, oracle=True, num_episide = num_ep)
