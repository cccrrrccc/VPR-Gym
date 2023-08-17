# VTR-Gym

## Installation

1. Building VTR.
```
./install_apt_packages.sh
make env
source .venv/bin/activate
pip install -r requirements.txt
make
```
2. Download Titan Benchmarks

Following the instruction [here](https://docs.verilogtorouting.org/en/latest/tutorials/titan_benchmarks/#titan-benchmarks-tutorial).

3. Install cppzmq.

Following the instruction in [cppzmq](https://github.com/zeromq/cppzmq).

4. Install required dependencies for VPR-Gym
```
cd ./VprGym
pip install -r requirements.txt
```
5. Run Example
```
cd ./VprGym
python3 example.py
```

It is important to use the VprGym as the RL agent's working directory.
## Introduction
VTR-Gym project is a platform for exploring AI techniques in FPGA placement optimization. VTR-Gym connects [OpenAI Gym](https://www.gymlibrary.dev/) and [VTR](https://verilogtorouting.org/) in order to achieve seamless integration between  Python-based machine learning libraries and VTR, which allows researchers to focus on high-level algorithm design and reduces the engineering efforts required for transplanting ML libraries from Python to C++.

OpenAI Gym is a toolkit for reinforcement learning (RL) widely used in research. It provides a simple and standard way to represent an RL problem.

The Verilog to Routing (VTR) project is an open-source framework for conducting FPGA architecture and CAD research and development.

## Examples
We provide an example at [example.py](./VprGym/example.py)
## Basic Interface
1. Example Python script.
```
from src.vprGym import VprEnv, VprEnv_blk_type
import Agent

env = VprEnv()
agent = Agent.Agent()
done = False

while (done == False):
  action = agent.get_action()
  _, reward, done, info = env.step(action)
  agent.update(reward)

env.close()
```
2. Parallel processing environments
```
from src.vprGym import VprEnv, VprEnv_blk_type
import Agent
import os

# In order to parallelly run two or more environments
# It is necessary to assign different port numbers for communication purposes
env1 = VprEnv(port = '5555')
agent1 = Agent.Agent()
env2 = VprEnv_blk_type(port = '6666')
agent2 = Agent.Agent()

pid = os.fork()
if pid > 0:
  train(env1, agent1)
else:
  train(env2, agent2)

```
Please refer to our paper for a detailed description.
## How to Cite
The following paper may be used as a general citation for VTR:

Bibtex:
```
```


