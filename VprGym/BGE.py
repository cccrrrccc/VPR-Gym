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
	env = VprEnv(inner_num = float(inner_num), port = ip, seed = int(seed), directory = 'BGE_' + name +'_' + str(g), benchmark = direct, reward_func = 'basic')
	
	arm_to_feature = list(np.arange(env.num_actions))
	#arm_to_feature = create_arm_feature(env.num_actions, env.num_types)
	agent = Policies.BoltzmannGumbel(nbArms = len(arm_to_feature), C = float(g))
	done = False
	max_reward = 0
	rewards = []
	while (done == False):
		prediction = agent.choice()
		action = arm_to_feature[prediction]
		_, reward, done, info = env.step(action)
		
		if info == 'stage2':
			arm_to_feature = list(np.arange(env.num_actions))
			agent = Policies.BoltzmannGumbel(nbArms = len(arm_to_feature), C = float(g))
			continue
		#normalize the reward
		if (reward > max_reward):
			max_reward = reward
		if (max_reward != 0):
			reward = reward / max_reward
		rewards.append(reward)
		agent.getReward(prediction, reward)
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
	with open('BGE_' + g + '_' + name + '.log', 'w') as f:
		sys.stdout = f
		print(WLs)
		print(CPDs)
		print(RTs)
		
if __name__ == '__main__':
	directs = [
	'vtr_flow/benchmarks/titan_blif/gsm_switch_stratixiv_arch_timing.blif',
	'vtr_flow/benchmarks/titan_blif/bitonic_mesh_stratixiv_arch_timing.blif',
	'vtr_flow/benchmarks/titan_blif/mes_noc_stratixiv_arch_timing.blif',
	'vtr_flow/benchmarks/titan_blif/denoise_stratixiv_arch_timing.blif',
	'vtr_flow/benchmarks/titan_blif/cholesky_mc_stratixiv_arch_timing.blif'
	]
	names = ['gsm', 'bitonic', 'mes', 'denoise', 'cholesky_mc']
	g = sys.argv[1]
	ip = sys.argv[2]
	
	assert(len(directs) == len(names))
	
	for i in range(len(directs)):
		batch_train(directs[i], g, ip, names[i])
