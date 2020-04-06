import gym
from random import uniform

def run(env, w1, w2, render = False):
    observation = env.reset()
    total_reward = 0

    for t in range(200):
        if render:
            env.render()

        l1 = [sum([observation[y] * w1[x][y] for y in range(4)]) for x in range(4)]
        l1 = [l1[x] if l1[x] > 0 else 0 for x in range(4)]
        result = sum([l1[x] * w2[x] for x in range(4)])

        action = 0 if result < 0 else 1
        observation, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward

env = gym.make('CartPole-v0')

_w1 = None
_w2 = None
_reward = 0

for _ in range(10000):
    w1 = [[uniform(-1, 1) for x in range(4)] for y in range(4)]
    w2 = [uniform(-1, 1) for x in range(4)]

    reward = run(env, w1, w2)

    if reward > _reward:
        _w1 = w1
        _w2 = w2
        _reward = reward

        if reward == 200:
            break

print("Best w1:")
print(_w1)
print("Best w2:")
print(_w2)
print("Best reward: {}".format(_reward))

for _ in range(10):
    run(env, _w1, _w2, True)
