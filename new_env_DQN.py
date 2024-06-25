import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import random
from collections import deque
import itertools
import copy

from env_cfg import Config, TestDemand, Agent


def get_init_len(init): #  计算一个一维数组或嵌套列表中的元素总数。
    """
    Calculate total number of elements in a 1D array or list of lists.
    :type init: iterable, list or (list of lists)
    :rtype: int
    """
    is_init_array = all([isinstance(x, (float, int, np.int64)) for x in init])
    if is_init_array:
        init_len = len(init)
    else:
        init_len = len(list(itertools.chain.from_iterable(init)))
    return init_len

class BeerGame(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, n_agents=4, n_turns_per_game=100, test_mode=False):
        super().__init__()
        c = Config()
        config, unparsed = c.get_config()
        self.config = config
        self.test_mode = test_mode
        if self.test_mode:
            self.test_demand_pool = TestDemand()

        self.curGame = 1 # The number associated with the current game (counter of the game)
        self.curTime = 0
        self.m = 10             #window size
        self.totIterPlayed = 0  # total iterations of the game, played so far in this and previous games
        self.players = self.createAgent()  # create the agents
        self.T = 0
        self.demand = []
        self.orders = []
        self.shipments = []
        self.rewards = []
        self.cur_demand = 0

        self.ifOptimalSolExist = self.config.ifOptimalSolExist
        self.getOptimalSol()

        self.totRew = 0    # it is reward of all players obtained for the current player. 
        self.totalReward = 0
        self.n_agents = n_agents

        self.n_turns = n_turns_per_game
        seed  = random.randint(0,1000000)
        self.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.totalTotal = 0

        # Agent 0 has 5 (-2, ..., 2) + AO       
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5),gym.spaces.Discrete(5)]))

        ob_spaces = {}
        for i in range(self.m):
            ob_spaces[f'current_stock_minus{i}'] = spaces.Discrete(5)
            ob_spaces[f'current_stock_plus{i}'] = spaces.Discrete(5)
            ob_spaces[f'OO{i}'] = spaces.Discrete(5)
            ob_spaces[f'AS{i}'] = spaces.Discrete(5)
            ob_spaces[f'AO{i}'] = spaces.Discrete(5)

        # Define the observation space, x holds the size of each part of the state
        x = [750, 750, 170, 45, 45]
        oob = []
        for _ in range(self.m):
          for ii in range(len(x)):
            oob.append(x[ii])
        self.observation_space = gym.spaces.Tuple(tuple([spaces.MultiDiscrete(oob)] * 4))

        # print("Observation space:")
        # print(self.observation_space)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def createAgent(self):
      agentTypes = self.config.agentTypes
      return [Agent(i,self.config.ILInit[i], self.config.AOInit, self.config.ASInit[i],
                                self.config.c_h[i], self.config.c_p[i], self.config.eta[i],
                                agentTypes[i],self.config) for i in range(self.config.NoAgent)]


    def resetGame(self, demand, ):
        self.demand = demand
        self.playType='test'
        self.curTime = 0
        self.curGame += 1
        self.totIterPlayed += self.T
        self.T = self.planHorizon()         #now fixed
        self.totalReward = 0

        self.deques = []
        for i in range(self.n_agents):
            deques = {}
            deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
            self.deques.append(deques)

        # reset the required information of player for each episode
        for k in range(0,self.config.NoAgent):
            self.players[k].resetPlayer(self.T)

        # update OO when there are initial IL,AO,AS
        self.update_OO()


    def reset(self):
        if self.test_mode:
            demand = self.test_demand_pool.next()
            if not self.test_demand_pool:           #if run out of testing data
                self.test_demand_pool = TestDemand()
        else:
            demand = [random.randint(0,2) for _ in range(102)]
            
        self.resetGame(demand)
        observations = [None] * self.n_agents

        self.deques = []
        for i in range(self.n_agents):
            deques = {}
            deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
            self.deques.append(deques)

        # prepend current observation
        # get current observation, prepend to deque
        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
            self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
            self.deques[i]['OO'].appendleft(int(curState[2]))
            self.deques[i]['AS'].appendleft(int(curState[3]))
            self.deques[i]['AO'].appendleft(int(curState[4]))

        # return entire m observations
        obs = [[], [], [], []]
        for i in range(self.n_agents):
            spaces = {}
            for j in range(self.m):
                obs[i].append(self.deques[i]['current_stock_minus'][j])
                obs[i].append(self.deques[i]['current_stock_plus'][j])
                obs[i].append(self.deques[i]['OO'][j])
                obs[i].append(self.deques[i]['AS'][j])
                obs[i].append(self.deques[i]['AO'][j])
                # spaces[f'current_stock_minus{j}'] = self.deques[i]['current_stock_minus'][j]
                # spaces[f'current_stock_plus{j}'] = self.deques[i]['current_stock_plus'][j]
                # spaces[f'OO{j}'] = self.deques[i]['OO'][j]
                # spaces[f'AS{j}'] = self.deques[i]['AS'][j]
                # spaces[f'AO{j}'] = self.deques[i]['AO'][j]

            # observations[i] = spaces

        obs_array = np.array([np.array(row) for row in obs])
        return obs_array  # observations #self._get_observations()


    def step(self, action:list):
        if get_init_len(action) != self.n_agents:
            raise error.InvalidAction(f'Length of action array must be same as n_agents({self.n_agents})')
        if any(np.array(action) < 0):
            raise error.InvalidAction(f"You can't order negative amount. You agents actions are: {action}")

        self.handleAction(action)
        self.next()

        self.orders = action

        for i in range(self.n_agents):
            self.players[i].getReward()
        self.rewards = [1 * self.players[i].curReward for i in range(0, self.config.NoAgent)]

        if self.curTime == self.T+1:
            self.done = [True] * 4
        else:
            self.done = [False] * 4


        # get current observation, prepend to deque
        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
            self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
            self.deques[i]['OO'].appendleft(int(curState[2]))
            self.deques[i]['AS'].appendleft(int(curState[3]))
            self.deques[i]['AO'].appendleft(int(curState[4]))

        # return entire m observations
        obs = [[],[],[],[]]
        observations = [None] * self.n_agents
        for i in range(self.n_agents):
            spaces = {}
            for j in range(self.m):
              obs[i].append(self.deques[i]['current_stock_minus'][j])
              obs[i].append(self.deques[i]['current_stock_plus'][j])
              obs[i].append(self.deques[i]['OO'][j])
              obs[i].append(self.deques[i]['AS'][j])
              obs[i].append(self.deques[i]['AO'][j])

        obs_array = np.array([np.array(row) for row in obs])
        state = obs_array #observations #self._get_observations()
        return state, self.rewards, self.done, {}



    def handleAction(self, action):
        # get random lead time
        leadTime = random.randint(self.config.leadRecOrderLow[0], self.config.leadRecOrderUp[0])
        self.cur_demand = self.demand[self.curTime]
        # set AO
        BS = False
        self.players[0].AO[self.curTime] += self.demand[self.curTime]       #orders from customer, add directly to the retailer arriving order
        for k in range(0, self.config.NoAgent):
            if k >= 0:  #recording action
                self.players[k].action = np.zeros(5)        #one-hot transformation
                self.players[k].action[action[k]] = 1
                BS = False
            else:
                raise NotImplementedError
                self.getAction(k)
                BS = True

            # updates OO and AO at time t+1
            self.players[k].OO += self.players[k].actionValue(self.curTime, self.playType, BS = BS)     #open order level update
            leadTime = random.randint(self.config.leadRecOrderLow[k], self.config.leadRecOrderUp[k])        #order
            if self.players[k].agentNum < self.config.NoAgent-1:
                if k>=0:
                    self.players[k + 1].AO[self.curTime + leadTime] += self.players[k].actionValue(self.curTime,
                                                                                                   self.playType,
                                                                                                   BS=False)  # TODO(yan): k+1 arrived order contains my own order and the order i received from k-1
                else:
                    raise NotImplementedError
                    self.players[k + 1].AO[self.curTime + leadTime] += self.players[k].actionValue(self.curTime,
                                                                                                   self.playType,
                                                                                                   BS=True)  # open order level update

    def next(self):
        # get a random leadtime for shipment
        leadTimeIn = random.randint(self.config.leadRecItemLow[self.config.NoAgent - 1],
                                    self.config.leadRecItemUp[self.config.NoAgent - 1])

        # handle the most upstream recieved shipment
        self.players[self.config.NoAgent-1].AS[self.curTime + leadTimeIn] += self.players[self.config.NoAgent-1].actionValue(self.curTime, self.playType, BS=True)
                                                                #the manufacture gets its ordered beer after leadtime

        self.shipments = []
        for k in range(self.config.NoAgent-1,-1,-1): # [3,2,1,0]

            # get current IL and Backorder
            current_IL = max(0, self.players[k].IL)
            current_backorder = max(0, -self.players[k].IL)

            # increase IL and decrease OO based on the action, for the next period
            self.players[k].recieveItems(self.curTime)

            # observe the reward
            possible_shipment = min(current_IL + self.players[k].AS[self.curTime],
                                    current_backorder + self.players[k].AO[self.curTime])       #if positive IL, ship all beer or all they needs, if backorders, ship all k-1 needs
            self.shipments.append(possible_shipment)

            # plan arrivals of the items to the downstream agent
            if self.players[k].agentNum > 0:
                leadTimeIn = random.randint(self.config.leadRecItemLow[k-1], self.config.leadRecItemUp[k-1])
                self.players[k-1].AS[self.curTime + leadTimeIn] += possible_shipment

            # update IL
            self.players[k].IL -= self.players[k].AO[self.curTime]

            # observe the reward
            self.players[k].getReward()
            rewards = [-1 * self.players[i].curReward for i in range(0, self.config.NoAgent)]

            # update next observation
            self.players[k].nextObservation = self.players[k].getCurState(self.curTime + 1)

        if self.config.ifUseTotalReward:  # default is false
            # correction on cost at time T
            if self.curTime == self.T:
                self.getTotRew()

        self.curTime += 1

    def getAction(self, k):
        self.players[k].action = np.zeros(self.config.actionListLenOpt)

        if self.config.demandDistribution == 2:
            if self.curTime and self.config.use_initial_BS <= 4:
                self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                        max(0, (self.players[k].int_bslBaseStock - (
                                                                    self.players[k].IL + self.players[k].OO -
                                                                    self.players[k].AO[self.curTime])))))] = 1
            else:
                self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                        max(0, (self.players[k].bsBaseStock - (
                                                                    self.players[k].IL + self.players[k].OO -
                                                                    self.players[k].AO[self.curTime])))))] = 1
        else:
            self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                                                    max(0, (self.players[k].bsBaseStock - (
                                                                self.players[k].IL + self.players[k].OO -
                                                                self.players[k].AO[self.curTime])))))] = 1

    def getTotRew(self):
      totRew = 0
      for i in range(self.config.NoAgent):
        # sum all rewards for the agents and make correction
        totRew += self.players[i].cumReward

      for i in range(self.config.NoAgent):
        self.players[i].curReward += self.players[i].eta*(totRew - self.players[i].cumReward) #/(self.T)


    def planHorizon(self):
      # TLow: minimum number for the planning horizon # TUp: maximum number for the planning horizon
      #output: The planning horizon which is chosen randomly.
      return random.randint(self.n_turns, self.n_turns)# self.config.TLow,self.config.TUp)

    def update_OO(self):
        for k in range(0,self.config.NoAgent):
            if k < self.config.NoAgent - 1:
                self.players[k].OO = sum(self.players[k+1].AO) + sum(self.players[k].AS)
            else:
                self.players[k].OO = sum(self.players[k].AS)

    def getOptimalSol(self):     
        # if self.config.NoAgent !=1:
        if self.config.NoAgent != 1 and 1 == 2:     
            # check the Shang and Song (2003) condition.
            for k in range(self.config.NoAgent - 1):
                if not (self.players[k].c_h == self.players[k + 1].c_h and self.players[k + 1].c_p == 0):
                    self.ifOptimalSolExist = False

            # if the Shang and Song (2003) condition satisfied, it runs the algorithm
            if self.ifOptimalSolExist == True:
                calculations = np.zeros((7, self.config.NoAgent))
                for k in range(self.config.NoAgent):
                    # DL_high
                    calculations[0][k] = ((self.config.leadRecItemLow + self.config.leadRecItemUp + 2) / 2 \
                                          + (self.config.leadRecOrderLow + self.config.leadRecOrderUp + 2) / 2) * \
                                         (self.config.demandUp - self.config.demandLow - 1)
                    if k > 0:
                        calculations[0][k] += calculations[0][k - 1]
                    # probability_high
                    nominator_ch = 0
                    low_denominator_ch = 0
                    for j in range(k, self.config.NoAgent):
                        if j < self.config.NoAgent - 1:
                            nominator_ch += self.players[j + 1].c_h
                        low_denominator_ch += self.players[j].c_h
                    if k == 0:
                        high_denominator_ch = low_denominator_ch
                    calculations[2][k] = (self.players[0].c_p + nominator_ch) / (
                                self.players[0].c_p + low_denominator_ch + 0.0)
                    # probability_low
                    calculations[3][k] = (self.players[0].c_p + nominator_ch) / (
                                self.players[0].c_p + high_denominator_ch + 0.0)
                # S_high
                calculations[4] = np.round(np.multiply(calculations[0], calculations[2]))
                # S_low
                calculations[5] = np.round(np.multiply(calculations[0], calculations[3]))
                # S_avg
                calculations[6] = np.round(np.mean(calculations[4:6], axis=0))
                # S', set the base stock values into each agent.
                for k in range(self.config.NoAgent):
                    if k == 0:
                        self.players[k].bsBaseStock = calculations[6][k]

                    else:
                        self.players[k].bsBaseStock = calculations[6][k] - calculations[6][k - 1]
                        if self.players[k].bsBaseStock < 0:
                            self.players[k].bsBaseStock = 0
        elif self.config.NoAgent == 1:
            if self.config.demandDistribution == 0:
                self.players[0].bsBaseStock = np.ceil(
                    self.config.c_h[0] / (self.config.c_h[0] + self.config.c_p[0] + 0.0)) * ((
                                                                                                         self.config.demandUp - self.config.demandLow - 1) / 2) * self.config.leadRecItemUp
        elif 1 == 1:
            f = self.config.f
            f_init = self.config.f_init
            for k in range(self.config.NoAgent):
                self.players[k].bsBaseStock = f[k]
                self.players[k].int_bslBaseStock = f_init[k]

    def render(self, mode='human'):
        # if mode != 'human':
        #     raise NotImplementedError(f'Render mode {mode} is not implemented yet')
        # print("")
        # print('\n' + '=' * 20)
        # print('Turn:     ', self.curTime)
        stocks = [p.IL for p in self.players]
        # print('Stocks:   ', ", ".join([str(x) for x in stocks]))
        # print('Orders:   ', self.orders)
        # print('Shipments:', self.shipments)
        # print('Rewards:', self.rewards)
        # print('Customer demand: ', self.cur_demand)

        AO = [p.AO[self.curTime] for p in self.players]
        AS = [p.AS[self.curTime] for p in self.players]

        # print('Arrived Order: ', AO)
        # print('Arrived Shipment: ', AS)

        OO = [p.OO for p in self.players]
        # print('Working Order: ', OO)


        # print('Last incoming orders:  ', self.next_incoming_orders)
        # print('Cum holding cost:  ', self.cum_stockout_cost)
        # print('Cum stockout cost: ', self.cum_holding_cost)
        # print('Last holding cost: ', self.holding_cost)
        # print('Last stockout cost:', self.stockout_cost)

        return stocks  

import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    
class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        # print("states shape:", states.shape)
        # print("actions shape:", actions.shape)
        # print("self.q_net(states) 输出:", self.q_net(states))

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            # print("q net state: ", self.q_net(state))
            action = self.q_net(state).argmax().item()
        return action

def basestock(stocks, s1, s2):
    if stocks < s1:
        a = min(s2-stocks, 4)
    else:
        a = 0
    return a



if __name__ == "__main__":
    agent_i = 1
    s1 = 0
    s2 = 6
    env = BeerGame()

    #  DQN setting
    lr = 2e-3
    num_episodes = 1200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    state_dim = 50
    action_dim = 5      #  维度 
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    return_list_1 = []
    return_list_2 = []
    for i_episode in range(num_episodes):
        episode_return_1 = 0
        episode_return_2 = 0
        if i_episode % 10 == 0:
            print('episode = ', i_episode)      #  记录当前的episode
        obs = env.reset()
        next_obs = copy.deepcopy(obs)  #  easy to change
        stock = env.render()        #  取stock
        done = False
        while not done:
            # rnd_action = env.action_space.sample()
            rnd_action = [0, 0, 0, 0]
            action = copy.deepcopy(rnd_action)
            state = list(next_obs[agent_i -1])    #  定义state
            # state = list(range(1, 51))
            # print('state: ', state)
            for i in range(len(rnd_action)):
                if i == (agent_i -1) or i == (agent_i +1):
                    action[i] = agent.take_action(state)    
                    action_i = action[i]        #  dqn决策的action
                else:
                    action[i] = basestock(stock[i], s1, s2)
            next_obs, reward, done_list, _ = env.step(tuple(action))
            done = all(done_list)
            stock = env.render()        #  取stock状态

            next_state = list(next_obs[agent_i -1])       #  定义next_state
            # next_state = list(range(1, 51))
            replay_buffer.add(state, action_i, reward[agent_i -1], next_state, done)
            state = next_state
            episode_return_1 += reward[agent_i-1]
            episode_return_2 += reward[agent_i]
            # 当buffer数据的数量超过一定值后,才进行Q网络训练
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)
        return_list_1.append(episode_return_1)
        return_list_2.append(episode_return_2)

    episodes_list_1 = list(range(len(return_list_1)))
    plt.plot(episodes_list_1, return_list_1)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on Beergame of MultiAgent '+ str(agent_i)+'1')
    # plt.show()
    plt.savefig('DQN on Beergame of MultiAgent '+str(agent_i)+'1'+'.png')

    mv_return = rl_utils.moving_average(return_list_1, 9)
    plt.plot(episodes_list_1, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('fig/DQN on Beergame of MultiAgent'+ str(agent_i)+'1')
    # plt.show()
    plt.savefig('fig/DQN on Beergame of MultiAgent '+str(agent_i)+'1'+' smooth.png')

    import pandas as pd 
    pd.DataFrame(return_list_1).to_csv('data/DQN on Beergame of MultiAgent'+ str(agent_i)+'1'+'.csv')

    episodes_list_2 = list(range(len(return_list_2)))
    plt.plot(episodes_list_2, return_list_2)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on Beergame of MultiAgent '+ str(agent_i) + '3')
    # plt.show()
    plt.savefig('DQN on Beergame of MultiAgent '+str(agent_i) + '3'+'.png')

    mv_return = rl_utils.moving_average(return_list_2, 9)
    plt.plot(episodes_list_2, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('fig/DQN on Beergame of MultiAgent'+ str(agent_i) + '3')
    # plt.show()
    plt.savefig('fig/DQN on Beergame of MultiAgent '+str(agent_i) + '3'+' smooth.png')

    import pandas as pd 
    pd.DataFrame(return_list_2).to_csv('data/DQN on Beergame of MultiAgent'+ str(agent_i)+ '2'+'.csv')