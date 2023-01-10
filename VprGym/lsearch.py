from src.vprGym import VprEnv, VprEnv_blk_type
import random
import nevergrad as ng
import numpy as np
import matplotlib.pyplot as plt
import sys

def create_arm_feature(num_actions, num_types):
	arm_to_feature = []
	for i in range(num_actions):
		for j in range(num_types):
			arm_to_feature.append([i, j])
	return arm_to_feature

def train(inner_num, seed, direct, ip, name):
	np.random.seed(int(seed))
	env = VprEnv(inner_num = float(inner_num), port = ip, seed = int(seed), directory = 'Nevergrad_CMA' + name, benchmark = direct, reward_func = 'basic')
	
	arm_to_feature = list(np.arange(env.num_actions))
	#arm_to_feature = create_arm_feature(env.num_actions, env.num_types)
	instrum = ng.p.Array(init=[0] * env.num_actions)
	optimizer = ng.optimizers.CMA(parametrization=instrum)
	done = False
	max_reward = 0
	acc_reward = 0
	G = 0
	count = 0
	loss = 0
	while (done == False):
		if count == 0:
			Q = optimizer.ask()
			#prob = np.exp(Q.value)
			prob = 1 / (1 + np.exp(-1 * Q.value)) # sigmoid
		prediction = random.choices(list(np.arange(len(arm_to_feature))), weights = list(prob))[0]
		_, reward, done, info = env.step(prediction)
		
		if info == 'stage2':
			arm_to_feature = list(np.arange(env.num_actions))
			instrum = ng.p.Array(init=[0] * env.num_actions)
			optimizer = ng.optimizers.CMA(parametrization=instrum)
			count = 0
			continue
		loss += -1 * reward
		count += 1
		# Batch size = 100
		if count == 100:
			optimizer.tell(Q, loss)
			loss = 0
			count = 0
	return info['WL'], info['CPD'], info['RT']
	
if __name__ == '__main__':
	direct = sys.argv[1]
	g = sys.argv[2] # default 0.9
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
	with open('Nevergrad_CMA_' + name + '.log', 'w') as f:
		sys.stdout = f
		print(WLs)
		print(CPDs)
		print(RTs)
