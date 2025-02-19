import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                 dqn_type='VanllaDQN'):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def save(self,path,epoch):
        torch.save(self.q_net.state_dict(),path+"/qnet_"+str(epoch)+".pt")

    def load(self,path,epoch):
        self.q_net.load_state_dict(torch.load(path+"/qnet_"+str(epoch)+".pt"))

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).argmax().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_value = self.q_net(states).gather(1, actions)
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net(states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        temp = torch.unsqueeze(rewards, dim=1)
        q_targets = temp + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_value, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

# def dis_to_con(discrete_action, env, action_dim):
#         action_lowbound = env.action_space.low[0]
#         action_upbound = env.action_space.high[0]
#         return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)

def DQN_train(agent, env, num_episodes, replay_buffer, minmal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    states=[]
    dones=[]
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state,_ = env.reset()
                done = False
                if state[0]<21 or state[1]<2:
                    print("state start!",len(states))
                while not done:
                    states.append(state)
                    dones.append(int(done))
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
                    max_q_value_list.append(max_q_value)
                    # action_continuous = dis_to_con(action, env, agent.action_dim)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minmal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                states.append(state)
                dones.append(int(done))
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    np.savez("BoxBall_trajectry.npz",states=np.array(states),dones=np.array(dones))
    return return_list, max_q_value_list

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_DQN(env,env_name=None,path=None):
    learning_rate = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minmal_size = 1000
    batch_size = 64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # env_name = 'BoxBall-v0'
    # # env = gym.make(env_name, render_mode="human")
    # env =gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n





    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer =ReplayBuffer(buffer_size)
    # DQN newtwork train
    # agent=DQN(state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device)
    # double DQN network train
    agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                dqn_type='DoubleDQN')
    return_list, max_q_value_list = DQN_train(agent, env, num_episodes, replay_buffer, minmal_size, batch_size)
    agent.save(path,0)
    episodes_list = list(range(len(return_list)))
    mv_return =moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    # plt.axline((0,0),slope=len(max_q_value_list),c='orange',ls='--')
    # plt.axline((0,10),slope=len(max_q_value_list),c='red',ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title("DQN ON {}".format(env_name))
    plt.show()

def evlution_DQN(env,env_name=None,path=None,epochs=1):
    learning_rate = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minmal_size = 1000
    batch_size = 64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # env_name = 'BoxBall-v0'
    # # env = gym.make(env_name, render_mode="human")
    # env =gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    # double DQN network train
    agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                dqn_type='DoubleDQN')
    agent.load(path,0)
    states=[]
    dones=[]
    for i in range(epochs):
        state, _ = env.reset()
        done = False
        while not done:
            states.append(state)
            dones.append(int(done))
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        states.append(state)
        dones.append(int(done))

    np.savez("expert_Boxball_trajectry.npz",states=np.array(states),dones=np.array(dones))
