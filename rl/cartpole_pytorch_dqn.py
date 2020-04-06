import torch, gym, random

#Num    Observation             Min     	Max
#0      Cart Position 	        -2.4 	    2.4
#1 	    Cart Velocity 	        -Inf     	Inf
#2 	    Pole Angle 	            ~ -41.8 	~ 41.8
#3 	    Pole Velocity At Tip 	-Inf 	    Inf

gamma = 0.95
learning_rate = 0.01
exp = []

model = torch.nn.Sequential(
    torch.nn.Linear(4, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 2)
)

loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def run(env, model, exp):
    state = env.reset()
    done = False
    exp_start = len(exp)

    while not done:
        action = env.action_space.sample()
        state_next, reward, done, _ = env.step(action)
        exp.append([state, action, reward])
        state = state_next

    for i in xrange(len(exp) - 2, exp_start - 1, -1):
        exp[i][2] += gamma * exp[i + 1][2]

    return len(exp) - exp_start

def train(model, exp):
    samples = random.sample(exp, 200) if len(exp) >= 200 else exp

    for state, action, reward in samples:
        pred = model(torch.Tensor(state))
        real = pred.clone().detach()
        real[action] = reward
        loss = loss_fn(pred, real)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

env = gym.make('CartPole-v0')

for t in xrange(100):
    reward = run(env, model, exp)

for t in xrange(100):
    loss = train(model, exp)
    print("T = {}, loss = {}".format(t, loss))

print("Experience size: {}".format(len(exp)))

for _ in xrange(10):
    state = env.reset()
    done = False
    _reward = 0

    while not done:
        env.render()
        action = torch.Tensor.argmax(model(torch.Tensor(state))).item()
        state, reward, done, _ = env.step(action)
        _reward += reward

    print("Reward = {}".format(_reward))
