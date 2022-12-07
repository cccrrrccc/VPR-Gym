. venv/bin/activate
b1='vtr_flow/benchmarks/titan_blif/stereo_vision_stratixiv_arch_timing.blif'
b2='vtr_flow/benchmarks/titan_blif/bitonic_mesh_stratixiv_arch_timing.blif'
b3='vtr_flow/benchmarks/titan_blif/neuron_stratixiv_arch_timing.blif'
b4='vtr_flow/benchmarks/titan_blif/SLAM_spheric_stratixiv_arch_timing.blif'


python3 exp3ELM.py $b3 0.01 7777 neuron &
#python3 softmax.py $b3 0.01 7778 neuron &
wait
