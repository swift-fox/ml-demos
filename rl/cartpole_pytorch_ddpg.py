import torch, gym, random, collections

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

actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=learning_rate)
critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=learning_rate)
critic_loss_fn = torch.nn.MSELoss(reduction='sum')

exp = collections.deque(maxlen=1000000)

def run(env, actor_model, critic_model):
    states = []
    probs = []
    rewards = []
    done = False
    state = env.reset()

    while not done:
        state = torch.Tensor(state)
        states.append(state)

        pred = actor_model(state)
        m = torch.distributions.Categorical(pred)
        action = m.sample()
        state, reward, done, _ = env.step(action.item())

        probs.append(m.log_prob(action))
        rewards.append(reward)

    return (states, probs, rewards)

def train(states, probs, rewards, exp):
    discounted_rewards = []
    _r = 0

    # Experience data preparation
    for r in rewards[::-1]:
        _r = r + gamma * _r
        discounted_rewards.insert(0, _r)

    discounted_rewards = torch.Tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

    exp += zip(states, discounted_rewards)

    # Train the critic
    samples = random.sample(exp, 200) if len(exp) >= 200 else exp

    sample_states, sample_rewards = zip(*samples)
    sample_states = torch.stack(sample_states)
    sample_rewards = torch.stack(sample_rewards)

    sample_values = critic_model(sample_states)

    critic_loss = critic_loss_fn(sample_values, sample_rewards)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Train the actor
    states = torch.stack(states)
    probs = torch.stack(probs)
    values = critic_model(states)

    actor_loss = -probs.mul(discounted_rewards - values).sum()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return (actor_loss.item(), critic_loss.item())

env = gym.make('CartPole-v1')

for t in range(5000):
    states, probs, rewards = run(env, actor_model, critic_model)
    actor_loss, critic_loss = train(states, probs, rewards, exp)
    print("T = {}\tactor_loss = {}\tcritic_loss={}".format(t, actor_loss, critic_loss))

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
