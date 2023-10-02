import zmq
from subprocess import Popen, PIPE
from gym import Env
import numpy as np
from gym.spaces import Discrete, Box, Tuple
import random
import os
import time

from .reader import read_WL_CPD

# Vpr option
default_seed = 10
default_inner_num = 0.1
default_arch = 'vtr_flow/arch/titan/stratixiv_arch.timing.xml'
default_benchmark = 'vtr_flow/benchmarks/titan_blif/neuron_stratixiv_arch_timing.blif'
default_addr = "tcp://localhost:"
default_port = "5555"
default_directory = 'experiment'
default_vtr_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Create a folder under given directory and change the current working directory to it
def handle_directory(directory, seed, inner_num, blk_type, port):
	try:
		os.chdir(directory)
	except FileNotFoundError:
		os.mkdir(directory)
		os.chdir(directory)

	try:
		os.mkdir('seed_' + str(seed) + '_inner_num_' + str(inner_num) + '_RL_gym_placement_blk_type_' + blk_type)
		os.chdir('seed_' + str(seed) + '_inner_num_' + str(inner_num) + '_RL_gym_placement_blk_type_' + blk_type)
	except FileExistsError:
		os.chdir('seed_' + str(seed) + '_inner_num_' + str(inner_num) + '_RL_gym_placement_blk_type_' + blk_type)
		
def back_to_init_directory():
	os.chdir('..')
	os.chdir('..')
			

class VprEnv(Env):
	def __init__(self, vtr_root = default_vtr_root_path, seed = default_seed, inner_num = default_inner_num, arch = default_arch, benchmark = default_benchmark, addr = default_addr, directory = default_directory, port = default_port, reward_func = 'WLbiased_runtime_aware'):
		handle_directory(directory, seed, inner_num, 'off', port)
		
		# Socket setup
		self.ctx = zmq.Context()
		self.socket = self.ctx.socket(zmq.REP)
	
		process = Popen([os.path.join(vtr_root, 'vpr/vpr')
		, os.path.join(vtr_root, arch)
		, os.path.join(vtr_root, benchmark)
		, '--route_chan_width', '300'
		, '--pack', '--place', '--seed', str(seed), '--inner_num', str(inner_num)
		, '--RL_gym_placement', 'on'
		, '--RL_gym_placement_blk_type', 'off'
		, '--RL_gym_port', port
		, '--place_reward_fun', reward_func
		])
		self.addr = addr + port
		self.socket.connect(self.addr)
		msgs = self.socket.recv_multipart()
		self.num_actions = int(msgs[0].decode('utf-8'))
		self.num_types = int(msgs[1].decode('utf-8'))
		self.horizon = int(msgs[2].decode('utf-8'))
		self.num_blks = []
		for i in range(3, len(msgs)):
			self.num_blks.append(int(msgs[i].decode('utf-8')))
		
		## Set up OpenAI GYM
		self.action_space = Discrete(self.num_actions)
		self.observation_space = Box(low=np.array([0]), high=np.array([0]))
		self.state = 0
		self.stage2 = False
		
	def step(self, action):
		self.socket.send_string(str(action))
		msg = self.socket.recv()
		info = {}
		if (msg.decode('utf-8') == 'end'):
			done = True
			reward = 0
			self.socket.disconnect(self.addr)
			time.sleep(30)
			WL, CPD, RT, SWAP = read_WL_CPD()
			info['WL'] = WL
			info['CPD'] = CPD
			info['RT'] = RT
			info['SWAP'] = SWAP
			back_to_init_directory()
		elif (msg.decode('utf-8') == 'reset'):
			done = False
			info = 'reset'
			reward = 0
		elif (msg.decode('utf-8') == 'stage2'):
			done = False
			info = 'stage2'
			reward = 0
			self.change_stage()
		else:	
			# Receive the reward from VPR
			# Tokenize the message to get reward/delta/delta_bb/delta_time 
			done = False
			tokens = msg.decode('utf-8').split()
			assert(len(tokens) == 4)
			reward = float(tokens[0])
			info['delta'] = float(tokens[1])
			info['delta_bb'] = float(tokens[2])
			info['delta_time'] = float(tokens[3])
		return self.state, reward, done, info
		
	def render(self):
		pass

	def reset(self):
		pass
		
	def change_stage(self):
		self.num_actions = 7
		self.stage2 = True
		
class VprEnv_blk_type(Env):
	def __init__(self, vtr_root = default_vtr_root_path, seed = default_seed, inner_num = default_inner_num, arch = default_arch, benchmark = default_benchmark, addr = default_addr, directory = default_directory, port = default_port, reward_func = 'WLbiased_runtime_aware'):
		handle_directory(directory, seed, inner_num, 'on', port)	
		
		# Socket setup
		self.ctx = zmq.Context()
		self.socket = self.ctx.socket(zmq.REP)
		
		
		process = Popen([os.path.join(vtr_root, 'vpr/vpr')
		, os.path.join(vtr_root, arch)
		, os.path.join(vtr_root, benchmark)
		, '--route_chan_width', '300'
		, '--pack', '--place', '--seed', str(seed), '--inner_num', str(inner_num)
		, '--RL_gym_placement', 'on'
		, '--RL_gym_placement_blk_type', 'on'
		, '--RL_gym_port', port
		, '--place_reward_fun', reward_func
		])
		self.addr = addr + port
		self.socket.connect(self.addr)
		msgs = self.socket.recv_multipart()
		self.num_actions = int(msgs[0].decode('utf-8'))
		self.num_types = int(msgs[1].decode('utf-8'))
		self.horizon = int(msgs[2].decode('utf-8'))
		self.num_blks = []
		for i in range(3, len(msgs)):
			self.num_blks.append(int(msgs[i].decode('utf-8')))
		
		## Set up OpenAI GYM
		self.action_space = Tuple((Discrete(self.num_actions), Discrete(self.num_types)))
		self.observation_space = Box(low=np.array([0]), high=np.array([0]))
		self.state = 0
		
		self.stage2 = False
		self.reset_happened = False
		
	def step(self, action):
		self.socket.send_multipart([str(action[0]).encode('utf-8'), str(action[1]).encode('utf-8')])
		msg = self.socket.recv()
		info = {}
		if (msg.decode('utf-8') == 'end'):
			done = True
			reward = 0
			self.socket.disconnect(self.addr)
			time.sleep(30)
			WL, CPD, RT, SWAP = read_WL_CPD()
			info['WL'] = WL
			info['CPD'] = CPD
			info['RT'] = RT
			info['SWAP'] = SWAP
			back_to_init_directory()
		elif (msg.decode('utf-8') == 'reset'):
			done = False
			info = 'reset'
			self.reset_happened = True
			reward = 0
		elif (msg.decode('utf-8') == 'stage2'):
			done = False
			info = 'stage2'
			reward = 0
			self.change_stage()
		else:
			# Receive the reward from VPR
			# Tokenize the message to get reward/delta/delta_bb/delta_time 
			done = False
			tokens = msg.decode('utf-8').split()
			assert(len(tokens) == 4)
			reward = float(tokens[0])
			info['delta'] = float(tokens[1])
			info['delta_bb'] = float(tokens[2])
			info['delta_time'] = float(tokens[3])
		return self.state, reward, done, info
		
	def render(self):
		pass

	def reset(self):
		pass
		
	def change_stage(self):
		self.num_actions = 7
		self.stage2 = True
