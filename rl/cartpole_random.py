import gym
from random import uniform

def run_episode(env, param):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        result = sum([observation[x] * param[x] for x in range(4)])
        action = 0 if result < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break

    return totalreward

env = gym.make('CartPole-v0')

bestparams = None
bestreward = 0

for _ in range(10000):
    parameters = [uniform(-1, 1) for x in range(4)]
    reward = run_episode(env,parameters)
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            break

print(bestparams)
print(bestreward)

param = bestparams

for t in range(10):
    observation = env.reset()
    for _ in range(200):
        env.render()
        result = sum([observation[x] * param[x] for x in range(4)])
        action = 0 if result < 0 else 1
        observation, reward, done, info = env.step(action)
        if done:
            break
