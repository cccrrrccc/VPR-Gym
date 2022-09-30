import zmq
from subprocess import Popen, PIPE
from gym import Env
import numpy as np
from gym.spaces import Discrete, Box, Tuple
import random

# Vpr option
default_seed = 10
default_inner_num = 0.1
default_arch = '../vtr_flow/arch/titan/stratixiv_arch.timing.xml'
default_benchmark = '../vtr_flow/benchmarks/titan_blif/neuron_stratixiv_arch_timing.blif'
default_addr = "tcp://localhost:5555"

# Socket setup
ctx = zmq.Context()
socket = ctx.socket(zmq.REP)


class VprEnv(Env):
	def __init__(self, seed = default_seed, inner_num = default_inner_num, arch = default_arch, benchmark = default_benchmark, addr = default_addr):
		process = Popen(['../vpr/vpr'
		, arch
		, benchmark
		, '--route_chan_width', '300'
		, '--pack', '--place', '--seed', str(seed), '--inner_num', str(inner_num)
		, '--RL_gym_placement', 'on'
		, '--RL_gym_placement_blk_type', 'off'
		])
		self.addr = addr
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
		
class VprEnv_2(Env):
	def __init__(self, seed = default_seed, inner_num = default_inner_num, arch = default_arch, benchmark = default_benchmark, addr = default_addr):
		process = Popen(['../vpr/vpr'
		, arch
		, benchmark
		, '--route_chan_width', '300'
		, '--pack', '--place', '--seed', str(seed), '--inner_num', str(inner_num)
		, '--RL_gym_placement', 'on'
		, '--RL_gym_placement_blk_type', 'on'
		])
		self.addr = addr
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
