from src.vprGym import VprEnv, VprEnv_blk_type
import random
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
import numpy as np
import sys

def create_arm_feature(num_actions):
	arm_to_feature = list(np.arange(num_actions))
	return arm_to_feature

def create_arm_feature(num_actions, num_types):
	arm_to_feature = []
	for i in range(num_actions):
		for j in range(num_types):
			arm_to_feature.append([i, j])
	return arm_to_feature
	

if __name__ == '__main__':
	inner_num = sys.argv[1]
	seed = sys.argv[2]
	env = VprEnv_blk_type(inner_num = float(inner_num), port = "7777", seed = int(seed), directory = 'blk_type_softmax_0.0003')
	options = list(np.arange(env.num_actions * env.num_types))
	arm_to_feature = create_arm_feature(env.num_actions, env.num_types)
	softmax = MAB(arms=options, learning_policy=LearningPolicy.Softmax(tau=0.0003))
	softmax.fit(decisions=[0], rewards=[0])
	done = False
	while (done == False):
		prediction = softmax.predict()
		action = arm_to_feature[prediction]
		#action = random.randint(0, env.num_actions - 1)
		_, reward, done, _ = env.step(action)
		softmax.partial_fit([prediction], [reward])


'''
env = VprEnv(inner_num = 0.1, port = "5555", seed = 0)
options = list(np.arange(env.num_actions))
arm_to_feature = list(np.arange(env.num_actions))
softmax = MAB(arms=options, learning_policy=LearningPolicy.Softmax(tau=0.0001))
softmax.fit(decisions=[0], rewards=[0])
done = False
while (done == False):
	prediction = softmax.predict()
	action = arm_to_feature[prediction]
	#action = random.randint(0, env.num_actions - 1)
	_, reward, done, _ = env.step(action)
	softmax.partial_fit([prediction], [reward])
'''
