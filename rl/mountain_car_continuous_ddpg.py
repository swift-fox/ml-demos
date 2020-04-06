import torch, gym, random, collections

# Observation
# Num 	Observation 	Min 	Max
# 0 	position 	    -1.2 	0.6
# 1 	velocity 	    -0.07 	0.07

# Actions
# Num 	Action
# 0 	Push car to the left (negative value) or to the right (positive value)

gamma = 0.99
learning_rate = 3e-4

actor_model = torch.nn.Sequential(
    torch.nn.Linear(2, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
)

critic_model = torch.nn.Sequential(
    torch.nn.Linear(3, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
)

actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=learning_rate)
critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=learning_rate)
critic_loss_fn = torch.nn.MSELoss(reduction='sum')

exp = collections.deque(maxlen=1000000)

def run(env, actor_model):
    states = []
    actions = []
    rewards = []
    done = False
    state = env.reset()

    while not done:
        state = torch.Tensor(state)
        states.append(state)

        action = actor_model(state)
        state, reward, done, _ = env.step([action.item()])

        actions.append(action)
        rewards.append(reward)

    return (states, actions, rewards)

def train(states, actions, rewards, exp):
    discounted_rewards = []
    _r = 0

    # Experience data preparation
    for r in rewards[::-1]:
        _r = r + gamma * _r
        discounted_rewards.insert(0, _r)

    discounted_rewards = torch.Tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

    exp += zip(states, actions, discounted_rewards)

    # Train the critic
    samples = random.sample(exp, 200) if len(exp) >= 200 else exp

    s_states, s_actions, s_rewards = zip(*samples)
    s_state_actions = [torch.cat(state_action) for state_action in zip(s_states, s_actions)]
    s_state_actions = torch.stack(s_state_actions)
    s_rewards = torch.stack(s_rewards)

    s_values = critic_model(s_state_actions)

    critic_loss = critic_loss_fn(s_values, s_rewards)

    critic_optimizer.zero_grad()
    critic_loss.backward(retain_graph=True)
    critic_optimizer.step()

    # Train the actor
    states = torch.stack(states)
    actions = torch.stack(actions)
    state_actions = [torch.cat(state_action) for state_action in zip(states, actions)]
    state_actions = torch.stack(state_actions)
    values = critic_model(state_actions)

    actor_loss = -values.sum()
    #actor_loss = -actions.mul(discounted_rewards - values).sum() # TODO: this is wrong, use state action value

    actor_optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    actor_optimizer.step()

    return (actor_loss.item(), critic_loss.item())

env = gym.make('MountainCarContinuous-v0')

for t in range(1000):
    states, actions, rewards = run(env, actor_model)
    actor_loss, critic_loss = train(states, actions, rewards, exp)
    print("T = {}\tactor_loss = {}\tcritic_loss={}".format(t, actor_loss, critic_loss))

for _ in range(10):
    state = env.reset()
    done = False
    _reward = 0

    while not done:
        env.render()
        action = actor_model(torch.Tensor(state))
        state, reward, done, _ = env.step([action.item()])
        _reward += reward

    print("Reward = {}".format(_reward))

env.close()
