import zmq
from subprocess import Popen, PIPE
from gym import Env
import numpy as np
from gym.spaces import Discrete, Box
import random

# Vpr option
seed = 10
inner_num = 0.1
arch = '../vtr_flow/arch/titan/stratixiv_arch.timing.xml'
benchmark = '../vtr_flow/benchmarks/titan_blif/neuron_stratixiv_arch_timing.blif'
addr = "tcp://localhost:5555"

# Socket setup
ctx = zmq.Context()
socket = ctx.socket(zmq.REP)
socket.connect(addr)



class VprEnv(Env):
	def __init__(self):
		process = Popen(['../vpr/vpr'
		, '../vtr_flow/arch/titan/stratixiv_arch.timing.xml'
		, '../vtr_flow/benchmarks/titan_blif/neuron_stratixiv_arch_timing.blif'
		, '--route_chan_width', '300'
		, '--pack', '--place', '--seed', str(seed), '--inner_num', str(inner_num)
		, '--RL_gym_placement', 'on'
		])
		
		msg = socket.recv()
		self.num_actions = int(msg.decode('utf-8'))
		self.action_space = Discrete(int(msg.decode('utf-8')))
		self.observation_space = Box(low=np.array([0]), high=np.array([0]))
		self.state = 0
		
	def step(self, action):
		socket.send_string(str(action))
		msg = socket.recv()
		if (msg.decode('utf-8') == 'end'):
			done = True
			reward = 0
			socket.disconnect(addr)
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
	env = VprEnv()
	done = False
	while (done == False):
		action = random.randint(0, env.num_actions - 1)
		_, reward, done, _ = env.step(action)
	print("fin")
