from src.vprGym import VprEnv

env = VprEnv(inner_num = 0.2)
done = False
while (done == False):
	#action = [random.randint(0, env.num_actions - 1), random.randint(0, env.num_types - 1)]
	action = random.randint(0, env.num_actions - 1)
	_, reward, done, _ = env.step(action)
print("fin")
