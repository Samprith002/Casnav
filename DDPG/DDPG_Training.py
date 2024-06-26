import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import gym
from gym import spaces
import numpy as np

class CrowdNavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(CrowdNavigationEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.state = None
        self.goal = np.array([0.5, 0.5])
        self.reset()
        
    def reset(self):
        self.state = np.random.rand(4)
        return self.state
    
    def step(self, action):
        self.state[0] += action[0] * 0.1
        self.state[1] += action[1] * 0.1
        distance_to_goal = np.linalg.norm(self.state[:2] - self.goal)
        reward = -distance_to_goal
        done = distance_to_goal < 0.1
        return self.state, reward, done, {}
    
    def render(self, mode='human', close=False):
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Draw goal
        goal_circle = plt.Circle(self.goal, 0.03, color='green')
        ax.add_patch(goal_circle)

        # Draw robot
        robot_circle = plt.Circle(self.state[:2], 0.03, color='blue')
        ax.add_patch(robot_circle)

        # Draw obstacles (assuming self.state[2:] represents obstacles)
        obstacle_circle = plt.Circle(self.state[2:], 0.03, color='red')
        ax.add_patch(obstacle_circle)

        plt.title("Crowd Navigation Simulation")
        plt.show()
        
    def close(self):
        pass

from gym.envs.registration import register

register(
    id='CrowdNavigation-v0',
    entry_point='__main__:CrowdNavigationEnv',
)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)
    
    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        index = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in index]

    def size(self):
        return len(self.buffer)

def ddpg_update(actor, critic, target_actor, target_critic, buffer, batch_size, gamma, tau, actor_optimizer, critic_optimizer):
    batch = buffer.sample(batch_size)
    state, action, reward, next_state, done = zip(*batch)
    
    state = torch.FloatTensor(state)
    action = torch.FloatTensor(action)
    reward = torch.FloatTensor(reward).unsqueeze(1)
    next_state = torch.FloatTensor(next_state)
    done = torch.FloatTensor(done).unsqueeze(1)
    
    target_action = target_actor(next_state)
    target_q = target_critic(next_state, target_action)
    target_value = reward + (1 - done) * gamma * target_q
    critic_loss = nn.MSELoss()(critic(state, action), target_value.detach())
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    actor_loss = -critic(state, actor(state)).mean()
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

env = gym.make('CrowdNavigation-v0')

actor = Actor(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], max_action=env.action_space.high[0])
critic = Critic(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
target_actor = Actor(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], max_action=env.action_space.high[0])
target_critic = Critic(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

replay_buffer = ReplayBuffer()

num_episodes = 2
batch_size = 64
gamma = 0.99
tau = 0.005
rewards = []

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    
    while True:
        action = actor(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add((state, action, reward, next_state, done))
        
        state = next_state
        episode_reward += reward
        
        if replay_buffer.size() > batch_size:
            ddpg_update(actor, critic, target_actor, target_critic, replay_buffer, batch_size, gamma, tau, actor_optimizer, critic_optimizer)
        
        if done:
            break
    
    rewards.append(episode_reward)
    print(f"Episode {episode}, Reward: {episode_reward}")

    env.render()
    
   




