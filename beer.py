import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import numpy as np
import gym
from gym import spaces
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib
matplotlib.use("TkAgg")  # 使用TkAgg后端
import matplotlib.pyplot as plt
import rl_utils
import beer_game_env.envs.env
# 定义经验回放缓存
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# 定义简单的Q网络
hidden_dim=128
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

# 定义DQN智能体
class DQNAgent:
    def __init__(self, state_dim, action_dim, capacity=1000, batch_size=64, gamma=0.99, learning_rate=0.001):
        self.q_net = QNet(state_dim, action_dim)
        self.target_q_net = QNet(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.count = 0
        self.target_update = 10
    def take_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, 5)  # 五个离散动作
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            #print(state_tensor.shape)
            q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def update(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state, dtype=torch.float32)
        #print(state_batch.shape)
        action_batch = torch.tensor(batch.action).view(-1, 1)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).view(-1, 1)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).view(-1, 1)
        #print(state_batch.shape,action_batch.shape)
        q_values = self.q_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_q_net(next_state_batch).max(1)[0].view(-1, 1)
        #print(reward_batch.shape,next_q_values.shape,done_batch.shape)
        target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = F.mse_loss(q_values, target_q_values)
        #print(q_values[0])
        #print(target_q_values[0])
        #print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print(self.q_net.fc1.bias)
        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count+=1

# 在BeerGame环境中使用DQN智能体
env_name = 'BeerGame-v0'
env = gym.make(env_name,n_agents=4)
#state_dim = env.observation_space.shape[0]
state_dim=4
action_dim = 5
agent = DQNAgent(state_dim, action_dim)

# 训练DQN智能体
epsilon = 1  # 初始epsilon值
epsilon_decay = 0.99  # 每步epsilon的衰减因子
episode_list = []
reward_list = []
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
for episode in range(1000):
    env.reset()
    state = [env.players[k].IL for k in range(env.n_agents)]
    total_reward = [0,0,0,0]

    for step in range(50):  # 限制每个episode的步数
        action = agent.take_action(state, epsilon)
        #print(action)
        action = [1,action,2,3]
        #print(state,'\n',action)
        #print(env.players[0].AS[step],env.players[1].AS[step],env.players[2].AS[step],env.players[3].AS[step])
        next_state, reward, ifdone, _ = env.step(action)
        done=all(ifdone)
        next_state = [env.players[k].IL for k in range(env.n_agents)]
        #print(env.players[k].AS[0] for k in range(env.n_agents))
        #print("Action:", action)
        #print("Reward:", reward)
        agent.memory.add(state, action[1], next_state, reward[1], done)
        agent.update()
        #print(total_reward.shape)
        #print(reward.shape)
        for i in range(4):
            total_reward[i] += reward[i]
        state = next_state

        if done:
            break

    epsilon *= epsilon_decay  # 每个episode结束后衰减epsilon
    episode_list.append(episode+1)
    reward_list.append(total_reward[1])
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
#reward_list=rl_utils.moving_average(reward_list, 9)
plt.plot(episode_list, reward_list)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('DQN on {}'.format("BeerGame"))
plt.show()