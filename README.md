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
VTR-Gym project is a world-wide collaborative effort to provide an open-source framework for conducting FPGA architecture and CAD research and development.
The VTR design flow takes as input a Verilog description of a digital circuit, and a description of the target FPGA architecture.
It then performs:
  * Elaboration & Synthesis (ODIN II)
  * Logic Optimization & Technology Mapping (ABC)
  * Packing, Placement, Routing & Timing Analysis (VPR)

to generate FPGA speed and area results.
VTR includes a set of benchmark designs known to work with the design flow.

VTR can also produce [FASM](https://symbiflow.readthedocs.io/en/latest/fasm/docs/specification.html) to program some commercial FPGAs (via [Symbiflow](https://symbiflow.github.io/))

| Placement (carry-chains highlighted) | Critical Path |
| ------------------------------------ | ------------- |
| <img src="https://verilogtorouting.org/img/des90_placement_macros.gif" width="350"/> | <img src="https://verilogtorouting.org/img/des90_cpd.gif" width="350"/> |

| Logical Connections | Routing Utilziation |
| ------------------- | ------------------- |
| <img src="https://verilogtorouting.org/img/des90_nets.gif" width="350"/> | <img src="https://verilogtorouting.org/img/des90_routing_util.gif" width="350"/> |


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


