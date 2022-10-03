import zmq
from subprocess import Popen, PIPE
from gym import Env
import numpy as np
from gym.spaces import Discrete, Box, Tuple
import random
import os

# Vpr option
default_seed = 10
default_inner_num = 0.1
default_arch = 'vtr_flow/arch/titan/stratixiv_arch.timing.xml'
default_benchmark = 'vtr_flow/benchmarks/titan_blif/neuron_stratixiv_arch_timing.blif'
default_addr = "tcp://localhost:"
default_port = "5555"
default_directory = 'experiment'
default_vtr_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Socket setup
ctx = zmq.Context()
socket = ctx.socket(zmq.REP)

# Create a folder under given directory and change the current working directory to it
def handle_directory(directory, seed, inner_num, blk_type):
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
			

class VprEnv(Env):
	def __init__(self, vtr_root = default_vtr_root_path, seed = default_seed, inner_num = default_inner_num, arch = default_arch, benchmark = default_benchmark, addr = default_addr, directory = default_directory, port = default_port):
		handle_directory(directory, seed, inner_num, 'off')
	
		process = Popen([os.path.join(vtr_root, 'vpr/vpr')
		, os.path.join(vtr_root, arch)
		, os.path.join(vtr_root, benchmark)
		, '--route_chan_width', '300'
		, '--pack', '--place', '--seed', str(seed), '--inner_num', str(inner_num)
		, '--RL_gym_placement', 'on'
		, '--RL_gym_placement_blk_type', 'off'
		])
		self.addr = addr + port
		socket.connect(self.addr)
		msgs = socket.recv_multipart()
		self.num_actions = int(msgs[0].decode('utf-8'))
		self.action_space = Discrete(self.num_actions)
		self.observation_space = Box(low=np.array([0]), high=np.array([0]))
		self.state = 0
		
	def step(self, action):
		socket.send_string(str(action))
		msg = socket.recv()
		if (msg.decode('utf-8') == 'end'):
			done = True
			reward = 0
			socket.disconnect(self.addr)
		else:
			done = False
			reward = float(msg.decode('utf-8'))
		info = {}
		return self.state, reward, done, info
		
	def render(self):
		pass


	def reset(self):
		pass
		
class VprEnv_blk_type(Env):
	def __init__(self, vtr_root = default_vtr_root_path, seed = default_seed, inner_num = default_inner_num, arch = default_arch, benchmark = default_benchmark, addr = default_addr, directory = default_directory, port = default_port):
		handle_directory(directory, seed, inner_num, 'on')	
		
		process = Popen([os.path.join(vtr_root, 'vpr/vpr')
		, os.path.join(vtr_root, arch)
		, os.path.join(vtr_root, benchmark)
		, '--route_chan_width', '300'
		, '--pack', '--place', '--seed', str(seed), '--inner_num', str(inner_num)
		, '--RL_gym_placement', 'on'
		, '--RL_gym_placement_blk_type', 'on'
		])
		self.addr = addr + port
		socket.connect(self.addr)
		msgs = socket.recv_multipart()
		self.num_actions = int(msgs[0].decode('utf-8'))
		self.num_types = int(msgs[1].decode('utf-8'))
		self.action_space = Tuple((Discrete(self.num_actions), Discrete(self.num_types)))
		self.observation_space = Box(low=np.array([0]), high=np.array([0]))
		self.state = 0
		
	def step(self, action):
		socket.send_multipart([str(action[0]).encode('utf-8'), str(action[1]).encode('utf-8')])
		msg = socket.recv()
		if (msg.decode('utf-8') == 'end'):
			done = True
			reward = 0
			socket.disconnect(self.addr)
		else:
			done = False
			reward = float(msg.decode('utf-8'))
		info = {}
		return self.state, reward, done, info
		
	def render(self):
		pass


	def reset(self):
		pass
		
		
if __name__ == '__main__':
	env = VprEnv(inner_num = 0.2)
	done = False
	while (done == False):
		#action = [random.randint(0, env.num_actions - 1), random.randint(0, env.num_types - 1)]
		action = random.randint(0, env.num_actions - 1)
		_, reward, done, _ = env.step(action)
	print("fin")
