from tensorflow_DDQN import BrainDQN
import pickle
from collections import deque
actions = 40

brain = BrainDQN(actions)
with open("C:\\Users\\OUYANG\\thisComputer\\memory.json", "rb") as fp:
    brain.replayMemory = deque(pickle.load(fp))

i = 0
while True:
    i += 1
    brain.trainQNetwork()
    if i % 100 == 0:
        print("TIMESTEP", brain.session.run(brain.timeStep))
        print("The cost is: ", brain.costValue)
        print()

