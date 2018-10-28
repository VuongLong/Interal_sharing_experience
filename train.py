# -*- coding: utf-8 -*-
import tensorflow as tf      # Deep Learning library

from network import PGNetwork
from network import VNetwork
from network import ZNetwork

from constant import ACTION_SIZE
from multitask_policy import MultitaskPolicy
from env.terrain import Terrain
import argparse

NON = 1
SHARE = 2

def training(test_time, map_index, share_exp, oracle, num_episide, num_epoch):
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
	oracle_network = {}
	for i in range(env.num_task):	
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

	for i in range (env.num_task-1):
		for j in range(i+1,env.num_task):	
			oracle_network[i,j] = ZNetwork(
					state_size = env.size_m, 
					action_size = 2, 
					learning_rate = learning_rate,
					name = "oracle"+str(i)+"_"+str(i)
					)

	sess = tf.Session()
	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.1)
	#sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	if share_exp:
		if oracle: 
			writer = tf.summary.FileWriter("../plot/log/oracle"+"_"+str(num_episide))
		else:	
			writer = tf.summary.FileWriter("../plot/log/Share_samples"+"_"+str(num_episide))
	else:
		writer = tf.summary.FileWriter("../plot/log/Non"+"_"+str(num_episide))
	
	test_name =  "map_"+str(map_index) + "_test_" + str(test_time)	
	tf.summary.scalar(test_name + "/rewards", policy[0].mean_reward)
	write_op = tf.summary.merge_all()

	multitask_agent = MultitaskPolicy(
									map_index = map_index,
									policy = policy,
									value = value,
									oracle_network = oracle_network,
									writer = writer,
									write_op = write_op,
									num_epochs = num_epoch,
									gamma = 0.99,
									plot_model = 10000,
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
	parser = argparse.ArgumentParser()
	parser.add_argument("--share_type", type=int, default = 1)
	parser.add_argument("--num_epoch", type=int, default = 1000)
	parser.add_argument("--num_episode", nargs='+', type=int, default = [4, 8, 12, 16, 20, 24])
	args = parser.parse_args()

	map_index = 1
	for num_ep in args.num_episode:
		#if args.share_type == 1:
		training(test_time=num_ep, map_index=map_index, share_exp = True, oracle=True, num_episide = num_ep, num_epoch =args.num_epoch)
		#if args.share_type == 2:
		training(test_time=num_ep, map_index=map_index, share_exp = True, oracle=False, num_episide = num_ep, num_epoch =args.num_epoch)
		#if args.share_type == 3:
		training(test_time=num_ep, map_index=map_index, share_exp = False, oracle=False, num_episide =num_ep, num_epoch =args.num_epoch)


'''
python train.py --share_type 1
python train.py --share_type 2
python train.py --share_type 3 --num_epoch 2000 --num_episode 4
python train.py --share_type 3 --num_epoch 2000 --num_episode 8
python train.py --share_type 3 --num_epoch 2000 --num_episode 12
python train.py --share_type 3 --num_epoch 2000 --num_episode 16
python train.py --share_type 3 --num_epoch 2000 --num_episode 20
python train.py --share_type 3 --num_epoch 2000 --num_episode 24
'''
