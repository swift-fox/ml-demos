import torch, gym, random

#Num    Observation             Min     	Max
#0      Cart Position 	        -2.4 	    2.4
#1 	    Cart Velocity 	        -Inf     	Inf
#2 	    Pole Angle 	            ~ -41.8 	~ 41.8
#3 	    Pole Velocity At Tip 	-Inf 	    Inf

gamma = 0.9
learning_rate = 0.01

model = torch.nn.Sequential(
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 2),
    torch.nn.Softmax()
)

loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def run(env, model):
    state = env.reset()
    done = False
    probs = []
    rewards = []

    while not done:
        pred = model(torch.Tensor(state))
        c = torch.distributions.Categorical(pred)
        action = c.sample()
        state, reward, done, _ = env.step(action.item())

        probs.append(c.log_prob(action))
        rewards.append(reward)

    return (probs, rewards)

def train(model, probs, rewards):
    discounted_rewards = []
    _r = 0

    for r in rewards[::-1]:
        _r = r + gamma * _r
        discounted_rewards.append(_r)

    discounted_rewards.reverse()
    discounted_rewards = torch.Tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

    probs = torch.stack(probs)

    loss = -probs.mul(discounted_rewards).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

env = gym.make('CartPole-v1')

for t in range(1000):
    probs, rewards = run(env, model)
    loss = train(model, probs, rewards)
    print("T = {}, loss = {}".format(t, loss))

for _ in range(10):
    state = env.reset()
    done = False
    _reward = 0

    while not done:
        env.render()
        pred = model(torch.Tensor(state))
        c = torch.distributions.Categorical(pred)
        action = c.sample().item()
        state, reward, done, _ = env.step(action)
        _reward += reward

    print("Reward = {}".format(_reward))

env.close()
