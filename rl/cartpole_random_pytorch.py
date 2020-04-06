import torch, gym, copy
from random import uniform

def run(env, model, render = False):
    observation = env.reset()
    total_reward = 0

    for t in range(200):
        if render:
            env.render()

        action = 0 if model(torch.Tensor(observation)) < 0 else 1
        observation, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward

def randomize_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, -1, 1)
        torch.nn.init.zeros_(m.bias)

env = gym.make('CartPole-v0')

model = torch.nn.Sequential(
    #torch.nn.Linear(4, 4),
    #torch.nn.ReLU(),
    torch.nn.Linear(4, 1)
)

_model = None
_reward = 0

for _ in range(10000):
    model.apply(randomize_weights)

    reward = run(env, model)

    if reward > _reward:
        _model = copy.deepcopy(model)
        _reward = reward

        if reward == 200:
            break

print("Best reward: {}".format(_reward))

for _ in range(10):
    run(env, _model, True)
