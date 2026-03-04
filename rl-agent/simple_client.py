import argparse
from ns3gym import ns3env
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5555)
parser.add_argument('--simTime', type=float, default=1.0)
parser.add_argument('--stepTime', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

env = ns3env.Ns3Env(
    port=args.port,
    stepTime=args.stepTime,
    startSim=False,   # IMPORTANT: do not start ns-3
    simSeed=args.seed,
    simArgs={},
    debug=False
)

ob_space = env.observation_space
ac_space = env.action_space

print("Observation space:", ob_space, ob_space.dtype)
print("Action space:", ac_space, ac_space.dtype)

obs = env.reset()
done = False
step = 0

while not done:
    print("Step: ", step)
    print("---obs: ", obs)

    # sample random action
    if hasattr(ac_space, "n"):
        action = ac_space.sample()
    else:
        action = np.zeros(ac_space.shape, dtype=ac_space.dtype)

    print("---action: ", action)

    obs, reward, done, info = env.step(action)

    print("---obs, reward, done, info: ", obs, reward, done, info)
    step += 1

env.close()
print("Done")
