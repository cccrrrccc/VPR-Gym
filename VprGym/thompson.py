from src.vprGym import VprEnv, VprEnv_blk_type
import random
import SMPyBandits.Policies as Policies
import numpy as np
import matplotlib.pyplot as plt
import sys

def create_arm_feature(num_actions, num_types):
	arm_to_feature = []
	for i in range(num_actions):
		for j in range(num_types):
			arm_to_feature.append([i, j])
	return arm_to_feature

def train(inner_num, seed, direct, g, ip, name):
	np.random.seed(int(seed))
	env = VprEnv(inner_num = float(inner_num), port = ip, seed = int(seed), directory = 'tps_' + name +'_' + str(g), benchmark = direct)
	
	arm_to_feature = list(np.arange(env.num_actions))
	#arm_to_feature = create_arm_feature(env.num_actions, env.num_types)
	agent = Policies.DiscountedThompson(nbArms = len(arm_to_feature), gamma=float(g))
	done = False
	max_reward = 0
	rewards = []
	while (done == False):
		prediction = agent.choice()
		action = arm_to_feature[prediction]
		_, reward, done, info = env.step(action)
		
		if info == 'stage2':
			arm_to_feature = list(np.arange(env.num_actions))
			agent = Policies.DiscountedThompson(nbArms = len(arm_to_feature), gamma=float(g))
			continue
		#normalize the reward
		if (reward > max_reward):
			max_reward = reward
		if (max_reward != 0):
			reward = reward / max_reward
		rewards.append(reward)
		agent.getReward(prediction, reward)
	return info['WL'], info['CPD'], info['RT']
	
if __name__ == '__main__':
	direct = sys.argv[1]
	g = sys.argv[2] # default 0.99
	ip = sys.argv[3]
	name = sys.argv[4]
	
	WLs = []
	CPDs = []
	RTs = []
	
	for inner_num in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
		WL = 0
		CPD = 0
		RT = 0
		for seed in [0, 1, 2]:
			a, b, c = train(inner_num, seed, direct, g, ip, name)
			WL += a
			CPD += b
			RT += c
		WLs.append(WL / 3)
		CPDs.append(CPD / 3)
		RTs.append(RT / 3)
	with open('tps_' + gamma + '_' + name + '.log', 'w') as f:
		sys.stdout = f
		print(WLs)
		print(CPDs)
		print(RTs)
