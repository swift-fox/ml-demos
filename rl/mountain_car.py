import torch, gym, random, copy, collections

# Observation
# Num 	Observation 	Min 	Max
# 0 	position 	    -1.2 	0.6
# 1 	velocity 	    -0.07 	0.07

# Actions
# Num 	Action
# 0 	push left
# 1 	no push
# 2 	push right

gamma = 0.99
learning_rate = 0.001
ep = 0.3
exp = collections.deque(maxlen=200000)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 3),
)

#target_model = copy.deepcopy(model)

loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def run(env, model, exp):
    state = env.reset()
    done = False
    _max = -100

#    while not done:
    for i in xrange(20000):
        if random.random() < ep:
            action = env.action_space.sample()
        else:
            action = torch.argmax(model(torch.Tensor(state))).item()

        state_next, reward, done, _ = env.step(action)

        # Reward shaping
#        reward += state_next[0] + 0.5

        if state_next[0] > _max:
#            reward += 10
            _max = state_next[0]

        if state_next[0] >= 0.5:
#            reward += 100
            if state_next[0] >= 0.51:
                break

        exp.append([state, action, reward, state_next])
        state = state_next

#    return exp[exp_start][2]
#    return exp_start - len(exp)
    return _max

def train(model, exp):
    if len(exp) < 500:
        return

    samples = random.sample(exp, 500)

    states = []
    actions = []
    rewards = []
    states_next = []

    for state, action, reward, state_next in samples:
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        states_next.append(state_next)

    q_next = model(torch.Tensor(states_next))
#    optimizer.zero_grad()
#    q_next = target_model(torch.Tensor(states_next))

    q = model(torch.Tensor(states))
    q_target = q.clone().detach()

    for i in xrange(len(samples)):
        q_target[actions[i]] = rewards[i] + gamma * torch.max(q_next[i]).item()

    loss = loss_fn(q, q_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

env = gym.make('MountainCar-v0')

for t in xrange(200):
#    if t % 100 == 0:
#        target_model.load_state_dict(model.state_dict())

    reward = run(env, model, exp)
#    ep -= 1.0 / 400
    for e in xrange(10):
        loss = train(model, exp)
    print("T = {}, reward={}, loss={}, ep={}".format(t, reward, loss, ep))

print("Experience size: {}".format(len(exp)))

#model.eval()

for _ in xrange(10):
    state = env.reset()
    done = False
    _reward = 0

    while not done:
        env.render()
        action = torch.Tensor.argmax(model(torch.Tensor(state))).item()
#        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        _reward += reward

    print("Reward = {}".format(_reward))
