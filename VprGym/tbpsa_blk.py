from src.vprGym import VprEnv, VprEnv_blk_type
import random
import nevergrad as ng
import numpy as np
import matplotlib.pyplot as plt
import sys


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

def train(inner_num, seed, direct, gamma, ip, name):
	np.random.seed(int(seed))
	env = VprEnv_blk_type(inner_num = float(inner_num), port = ip, seed = int(seed), directory = 'TBPSA_BLK_' + name +'_' + str(g), benchmark = direct, reward_func = 'basic')
	
	arm_features = stage1_arm_feature(env.num_actions, env.num_types)
	#arm_to_feature = create_arm_feature(env.num_actions, env.num_types)
	instrum = ng.p.Array(init=[0] * len(arm_features))
	optimizer = ng.optimizers.TBPSA(parametrization=instrum)
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
		prediction = arm_features[random.choices(list(np.arange(len(arm_features))), weights = list(prob))[0]]
		_, reward, done, info = env.step(prediction)
		
		if info == 'stage2':
			arm_features = stage2_arm_feature(env.num_actions, env.num_types)
			instrum = ng.p.Array(init=[0] * len(arm_features))
			optimizer = ng.optimizers.TBPSA(parametrization=instrum)
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
	
def batch_train(direct, g, ip, name):
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
	with open('TBPSA_BLK_' + name + '.log', 'w') as f:
		sys.stdout = f
		print(WLs)
		print(CPDs)
		print(RTs)
		
if __name__ == '__main__':
	directs = [
	'vtr_flow/benchmarks/titan_blif/denoise_stratixiv_arch_timing.blif',
	'vtr_flow/benchmarks/titan_blif/cholesky_mc_stratixiv_arch_timing.blif',
	'vtr_flow/benchmarks/titan_blif/mes_noc_stratixiv_arch_timing.blif',
	]
	names = ['denoise', 'cholesky_mc', 'mes']
	g = sys.argv[1]
	ip = sys.argv[2]
	
	assert(len(directs) == len(names))
	
	for i in range(len(directs)):
		batch_train(directs[i], g, ip, names[i])
