from src.vprGym import VprEnv, VprEnv_blk_type
import random
import SMPyBandits.Policies as Policies
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

def stage1_arm_feature(num_actions, num_types):
	arm_to_feature = []
	for i in range(num_actions):
		for j in range(num_types):
			arm_to_feature.append([i, j])
	return arm_to_feature
	
def stage1_weights(num_actions, probs):
	weights = []
	for i in range(num_actions):
		for p in probs:
			weights.append(p)
	return np.array(weights)
			
def stage2_arm_feature(num_actions, num_types):
	arm_to_feature = []
	for i in range(5):
		for j in range(num_types):
			arm_to_feature.append([i, j])
	arm_to_feature.append([5, 0])
	arm_to_feature.append([6, 0])
	return arm_to_feature

def stage2_weights(num_actions, probs):
	weights = []
	for i in range(5):
		for p in probs:
			weights.append(p)
	weights.append(1)
	weights.append(1)
	return np.array(weights)
	
Explore_Prob = 0.1

def train(inner_num, seed, direct, g, ip, name):
	np.random.seed(int(seed))
	env = VprEnv_blk_type(inner_num = float(inner_num), port = ip, seed = int(seed), directory = 'exp3ELM_' + name + '_' + str(g), benchmark = direct)
	probs = [x / sum(env.num_blks) for x in env.num_blks]
	weights = stage1_weights(env.num_actions, probs)
	arm_features = stage1_arm_feature(env.num_actions, env.num_types)
	agent =  Policies.Exp3ELM(nbArms = len(arm_features), delta = float(g), unbiased = True)
	agent.weights = weights.copy()
	## Set the weights of blk_agent according to the number of each type
	sum_reward = 0
	done = False
	max_reward = -1000000
	min_reward = 1000000
	local_max_reward = 0
	rewards = []
	count = 0
	horizon = env.horizon
	while (done == False):
		if random.uniform(0, 1) >= Explore_Prob:
			prediction = agent.choice()
		else:
			while True:
				prediction = random.choices(list(np.arange(len(arm_features))), weights = weights)[0]
				if prediction in agent.availableArms:
					break
		action = arm_features[prediction]
		_, reward, done, info = env.step(action)
		if info == 'stage2':
			weights = stage2_weights(env.num_actions, probs)
			arm_features = stage2_arm_feature(env.num_actions, env.num_types)
			agent =  Policies.Exp3ELM(nbArms = len(arm_features), delta = float(g), unbiased = True)
			agent.weights = weights.copy()
			continue
		#normalize the reward
		if (reward > max_reward):
			max_reward = reward
		if (reward < min_reward):
			min_reward = reward
		if (max_reward <= min_reward):
			reward = 0
		else:
			reward = (reward - min_reward) / (max_reward - min_reward)
		if (reward > local_max_reward):
			local_max_reward = reward
		agent.getReward(prediction, reward)
		rewards.append(reward)
		sum_reward += reward
		count += 1
		if count >= horizon:
			avg_reward = sum_reward / count
			if local_max_reward < 0:
				agent =  Policies.Exp3ELM(nbArms = len(arm_features), delta = float(g), unbiased = True)
				agent.weights = weights.copy()
				print('reset agent')
			count = 0
			sum_reward = 0
			local_max_reward = 0
			#agent =  Policies.Exp3ELM(nbArms = len(arm_features), delta = float(g), unbiased = True)
			#agent.weights = weights
	return info['WL'], info['CPD'], info['RT']
	
if __name__ == '__main__':
	direct = sys.argv[1]
	g = sys.argv[2]
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
	with open('EXP3ELM_' + name + '.log','w') as f:
		sys.stdout = f
		print(WLs)
		print(CPDs)
		print(RTs)
	
	
