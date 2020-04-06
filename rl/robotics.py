import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DDPG, HER, TD3, SAC

env = gym.make('FetchReach-v1')

#model = DDPG('MlpPolicy', env)
model = HER('MlpPolicy', env, DDPG, goal_selection_strategy='final', n_sampled_goal=4)
model.learn(50000000)
model.save('./her_fetch_reach')

#model = HER.load('./her_fetch_reach', env=env)

for _ in range(100):
    obs = env.reset()
    state = None
    done = False
    _reward = 0

    while not done:
        env.render()
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        _reward += reward

    print("Reward = {}".format(_reward))

env.close()
