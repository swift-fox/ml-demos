import torch, gym, random

#Num    Observation             Min     	Max
#0      Cart Position 	        -2.4 	    2.4
#1 	    Cart Velocity 	        -Inf     	Inf
#2 	    Pole Angle 	            ~ -41.8 	~ 41.8
#3 	    Pole Velocity At Tip 	-Inf 	    Inf

gamma = 0.99
learning_rate = 3e-4

actor_model = torch.nn.Sequential(
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 2),
    torch.nn.Softmax()
)

critic_model = torch.nn.Sequential(
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
)

params = list(actor_model.parameters()) + list(critic_model.parameters())
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(params, lr=learning_rate)

def run(env, actor_model, critic_model):
    state = env.reset()
    done = False
    probs = []
    rewards = []
    values = []

    while not done:
        state = torch.Tensor(state)
        value = critic_model(state).squeeze()

        pred = actor_model(state)
        c = torch.distributions.Categorical(pred)
        action = c.sample()
        state, reward, done, _ = env.step(action.item())

        probs.append(c.log_prob(action))
        values.append(value)
        rewards.append(reward)

    return (probs, values, rewards)

def train(probs, values, rewards):
    discounted_rewards = []
    _r = 0
#    _v = torch.zeros(1)

#    for i in reversed(range(len(rewards))):
#        _r = rewards[i] + gamma * _v.item()
#        _v = values[i]
    for r in rewards[::-1]:
        _r = r + gamma * _r
        discounted_rewards.append(_r)

    discounted_rewards.reverse()
    discounted_rewards = torch.Tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

    probs = torch.stack(probs)
    values = torch.stack(values)

    actor_loss = -probs.mul(discounted_rewards - values).sum()
    critic_loss = loss_fn(values, discounted_rewards)
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

env = gym.make('CartPole-v1')

for t in range(10000):
    probs, values, rewards = run(env, actor_model, critic_model)
    loss = train(probs, values, rewards)
    print("T = {}, loss = {}".format(t, loss))

for _ in range(10):
    state = env.reset()
    done = False
    _reward = 0

    while not done:
        env.render()
        pred = actor_model(torch.Tensor(state))
        c = torch.distributions.Categorical(pred)
        action = c.sample().item()
        state, reward, done, _ = env.step(action)
        _reward += reward

    print("Reward = {}".format(_reward))

env.close()
