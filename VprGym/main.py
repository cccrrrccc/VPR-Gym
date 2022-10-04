from src.vprGym import VprEnv, VprEnv_blk_type
import random
import mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

env = VprEnv_blk_type(inner_num = 0.1, port = "6666")
done = False
while (done == False):
	action = [random.randint(0, env.num_actions - 1), random.randint(0, env.num_types - 1)]
	#action = random.randint(0, env.num_actions - 1)
	_, reward, done, _ = env.step(action)
print("fin")
