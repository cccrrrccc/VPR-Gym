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
We provide an example of the basic environment at [example.py](./VprGym/example.py). Besides, we provide another example of the block-type enabled environment at [blocktype_example.py](./VprGym/blocktype_example.py).
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
3. Environment attributes
```
num_actions: An integer shows the number of available directed moves.

num_types: An integer shows the number of logical block types in the targeted FPGA design

num_blks: A list consist of the number of blocks for each logical block type category

horizon: An integer shows the number of steps within one temperature in the Simulated Annealing (SA) Process. Note that some multi-armed bandit algorithms require the horizon information in order to better plan the exploration and exploitation

is_stage2: This is a Boolean variable showing the current stage of the SA process. As mentioned above, it would affect the number of available directed move types. Besides, "info == 'stage2'" returned by the step function can also be used as a flag for stage 2.

info['delta'], info['delta_bb'], info['delta_time']: the normalized change of the circuit cost, the normalized change of the bounding box (related to Wire Length) cost, and the normalized change of the timing cost.
```

## How to Cite
The following paper may be used as a general citation for VPR:

Bibtex:
```
@article{murray2020vtr,
  title={Vtr 8: High-performance cad and customizable fpga architecture modelling},
  author={Murray, Kevin E and Petelin, Oleg and Zhong, Sheng and Wang, Jia Min and Eldafrawy, Mohamed and Legault, Jean-Philippe and Sha, Eugene and Graham, Aaron G and Wu, Jean and Walker, Matthew JP and others},
  journal={ACM Transactions on Reconfigurable Technology and Systems (TRETS)},
  volume={13},
  number={2},
  pages={1--55},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```

To cite VPR-Gym:

Bibtex:
```
@INPROCEEDINGS{10296368,
  author={Chen, Ruichen and Lu, Shengyao and Elgammal, Mohamed A. and Chun, Peter and Betz, Vaughn and Niu, Di},
  booktitle={2023 33rd International Conference on Field-Programmable Logic and Applications (FPL)}, 
  title={VPR-Gym: A Platform for Exploring AI Techniques in FPGA Placement Optimization}, 
  year={2023},
  volume={},
  number={},
  pages={72-78},
  keywords={Measurement;Machine learning algorithms;Design automation;Reinforcement learning;Market research;Libraries;Complexity theory;Reinforcement Learning;FPGA;Placement;VPR;OpenAI Gym},
  doi={10.1109/FPL60245.2023.00018}}

```
