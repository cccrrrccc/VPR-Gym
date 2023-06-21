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
## Introduction
VTR-Gym project is a platform for exploring AI techniques in FPGA placement optimization. VTR-Gym connects [OpenAI Gym](https://www.gymlibrary.dev/) and [VTR](https://verilogtorouting.org/) in order to achieve seamless integration between  Python-based machine learning libraries and VTR, which allows researchers to focus on high-level algorithm design and reduces the engineering efforts required for transplanting ML libraries from Python to C++.

OpenAI Gym is a toolkit for reinforcement learning (RL) widely used in research. It provides a simple and standard way to represent an RL problem.

The Verilog to Routing (VTR) project is an open-source framework for conducting FPGA architecture and CAD research and development.


## Documentation
VTR's [full documentation](https://docs.verilogtorouting.org) includes tutorials, descriptions of the VTR design flow, and tool options.

Also check out our [additional support resources](SUPPORT.md).

## License
Generally most code is under MIT license, with the exception of ABC which is distributed under its own (permissive) terms.
See the [full license](LICENSE.md) for details.

## How to Cite
The following paper may be used as a general citation for VTR:

K. E. Murray, O. Petelin, S. Zhong, J. M. Wang, M. ElDafrawy, J.-P. Legault, E. Sha, A. G. Graham, J. Wu, M. J. P. Walker, H. Zeng, P. Patros, J. Luu, K. B. Kent and V. Betz "VTR 8: High Performance CAD and Customizable FPGA Architecture Modelling", ACM TRETS, 2020.

Bibtex:
```
@article{vtr8,
  title={VTR 8: High Performance CAD and Customizable FPGA Architecture Modelling},
  author={Murray, Kevin E. and Petelin, Oleg and Zhong, Sheng and Wang, Jai Min and ElDafrawy, Mohamed and Legault, Jean-Philippe and Sha, Eugene and Graham, Aaron G. and Wu, Jean and Walker, Matthew J. P. and Zeng, Hanqing and Patros, Panagiotis and Luu, Jason and Kent, Kenneth B. and Betz, Vaughn},
  journal={ACM Trans. Reconfigurable Technol. Syst.},
  year={2020}
}
```

## Download
For most users of VTR (rather than active developers) you should download the [latest official VTR release](https://verilogtorouting.org/download), which has been fully regression tested.

## Development
This is the development trunk for the Verilog-to-Routing project.
Unlike the nicely packaged releases that we create, you are working with code in a constant state of flux.
You should expect that the tools are not always stable and that more work is needed to get the flow to run.

For new developers, please follow the [quickstart guide](https://docs.verilogtorouting.org/en/latest/quickstart/).

We follow a feature branch flow, where you create a new branch for new code, test it, measure its Quality of Results, and eventually produce a pull request for review by other developers. Pull requests that meet all the quality and review criteria are then merged into the master branch by a developer with the authority to do so.

In addition to measuring QoR and functionality automatically on pull requests, we do periodic automated testing of the master using BuildBot, and the results can be viewed below to track QoR and stability.
* [Trunk Status](http://builds.verilogtorouting.org:8080/waterfall)
* [QoR Tracking](http://builds.verilogtorouting.org:8080/)

*IMPORTANT*: A broken build must be fixed at top priority. You break the build if your commit breaks any of the automated regression tests.

For additional information see the [developer README](README.developers.md).


