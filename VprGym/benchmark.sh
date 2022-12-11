. venv/bin/activate
b1='vtr_flow/benchmarks/titan_blif/stereo_vision_stratixiv_arch_timing.blif'
b2='vtr_flow/benchmarks/titan_blif/bitonic_mesh_stratixiv_arch_timing.blif'
b3='vtr_flow/benchmarks/titan_blif/neuron_stratixiv_arch_timing.blif'
b4='vtr_flow/benchmarks/titan_blif/SLAM_spheric_stratixiv_arch_timing.blif'


for i in 1 2 3
do
	python3 CR_exp3.py 0.1 1 $b1 $i 6665 stereo_vision &
	python3 CR_exp3.py 0.1 2 $b1 $i 7775 stereo_vision &
	python3 CR_exp3.py 0.1 3 $b1 $i 8885 stereo_vision &
	wait
done
