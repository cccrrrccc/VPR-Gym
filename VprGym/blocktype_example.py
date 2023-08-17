import numpy as np
from numpy.random import randint
from src.vprGym import VprEnv, VprEnv_blk_type


if __name__ == '__main__':
	# Create a new environment
	# VprEnv_blk_type is the custom VPR environment, whose action space is directed move type coupling with logical block type
	# The action space is represented by a list [i, j]. (i = 0, ..., num_actions - 1 and j = 0, ..., num_types - 1) 
	env = VprEnv_blk_type(inner_num = 0.1, port = '6666', seed = 1, arch = 'vtr_flow/arch/titan/stratixiv_arch.timing.xml', directory = 'example', benchmark = 'vtr_flow/benchmarks/titan_blif/stereo_vision_stratixiv_arch_timing.blif', reward_func = 'basic')
	num_directed_moves = env.num_actions
	num_block_types = env.num_types
	avail_arms = []
	for i in range(num_directed_moves):
		for j in range(num_block_types):
			avail_arms.append([i, j])
	# done indicates whether the RL process is terminated or not
	done = False
	while (done == False):
		action = avail_arms[randint(num_actions)] # Provide an action from agent, here random search is used as agent
		_, reward, done, info = env.step(action) # pass the action to environment via env.step()
		current_reward = reward # The reward can be used to update agent
		
		# Note that VPR Placement is a two-stage process
		# The number of action will change as the stage changes
		if info == 'stage2':
			num_actions = env.num_actions
			avail_arms = list(np.arange(env.num_actions))
			# Reset the agent here
			continue
			
	# Print the result
	print('Wire Length:', info['WL'])
	print('Critical Path Delay:', info['CPD'])
	print('Runtime:', info['RT'])

