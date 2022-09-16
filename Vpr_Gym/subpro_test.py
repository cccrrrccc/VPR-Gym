import zmq
from subprocess import Popen, PIPE
from gym import Env
import numpy as np
from gym.spaces import Discrete, Box

seed = 10
inner_num = 0.7
process = Popen(['../vpr/vpr'
		, '../vtr_flow/arch/titan/stratixiv_arch.timing.xml'
		, '../vtr_flow/benchmarks/titan_blif/neuron_stratixiv_arch_timing.blif'
		, '--route_chan_width', '300'
		, '--pack', '--place', '--seed', str(seed), '--inner_num', str(inner_num)
		, '--RL_gym_placement', 'off'
		])
