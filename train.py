# -*- coding: utf-8 -*-
import tensorflow as tf      # Deep Learning library
import time

from network import *

from env.constants import ACTION_SIZE, RESOLUTION
from multitask_policy import MultitaskPolicy
import argparse


def training(test_time, scene_name, num_task, share_exp, oracle, num_episodes, num_epochs, num_steps, learning_rate=0.005):
	tf.reset_default_graph()
	
	if share_exp:
		if oracle:
			network_name_scope = 'Combine_gradients'
		else:
			network_name_scope = 'Share_samples'
	else:
		network_name_scope = 'Non'
	
	networks = []
	oracle_network = {}
	for i in range(num_task):	
		net = A2C(name='A2C', 
				resolution=RESOLUTION, 
				action_size=4,
				history_size=1,
				entropy_coeff=0.01,
				value_function_coeff=0.5,
				max_gradient_norm=None,
				joint_loss=False,
				learning_rate=learning_rate,
				decay=1,
				reuse=True
				)
		net.set_lr_decay(learning_rate, num_epochs * num_episodes * num_steps)
		networks.append(net)
		
		print("\nInitialized network with {} trainable weights.".format(len(net.find_trainable_variables('A2C', True))))

	for i in range (num_task-1):
		for j in range(i+1,num_task):	
			oracle_network[i,j] = ZNetwork(
					state_size = 100, 
					action_size = 2, 
					learning_rate = learning_rate*5,
					name = "oracle"+str(i)+"_"+str(j)
					)

	sess = tf.Session()
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	sess.run(tf.global_variables_initializer())


	if share_exp:
		if oracle: 
			writer = tf.summary.FileWriter("plot/log/oracle"+"_"+str(num_episodes))
		else:	
			writer = tf.summary.FileWriter("plot/log/Share_samples"+"_"+str(num_episodes))
	else:
		writer = tf.summary.FileWriter("plot/log/Non"+"_"+str(num_episodes))
	
	test_name =  "map_"+str(scene_name) + "_test_" + str(test_time)	
	pretrain_dir =  "pretrain/map"+str(scene_name) + "/"

	tf.summary.scalar(test_name + "/rewards", networks[0].total_mean_reward)
	tf.summary.scalar(test_name + "/redundant", networks[0].average_mean_redundant)

	for task_idx in range(num_task):
		tf.summary.scalar(test_name + "/rewards_task_{}".format(task_idx), networks[task_idx].mean_reward)
		tf.summary.scalar(test_name + "/redundant_task_{}".format(task_idx), networks[task_idx].mean_redundant)
	write_op = tf.summary.merge_all()

	multitask_agent = MultitaskPolicy(
									scene_name = scene_name,
									networks = networks,
									oracle_network = oracle_network,
									writer = writer,
									write_op = write_op,
									gamma = 0.99,
									plot_model = 20000,
									save_model = 100,
									num_task = num_task,
									num_epochs = num_epochs,
									num_episodes = num_episodes,
									num_steps = num_steps,
									share_exp = share_exp,
									oracle = oracle
									)
	
	multitask_agent.train(sess, pretrain_dir)
	sess.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_epoch", type=int, default = 1000)
	parser.add_argument("--num_step", type=int, default = 100)
	parser.add_argument("--num_episode", nargs='+', type=int, default = [8,16,24])
	
	args = parser.parse_args()

	scene_name = "Pygame_10x10"
	start = time.time()
	for num_step in [50, 100, 150]:
		training(test_time=num_step, scene_name=scene_name, num_task=1, share_exp=False, oracle=False, num_episodes=24, num_epochs=args.num_epoch, num_steps=num_step)
		# training(test_time=num_ep, scene_name=scene_name, share_exp = True, oracle=True, num_episode = num_ep, num_epoch =args.num_epoch)
		# training(test_time=num_ep, scene_name=scene_name, share_exp = True, oracle=False, num_episode = num_ep, num_epoch =args.num_epoch)

	print("Elapsed time: {}".format((time.time() - start)/3600))
