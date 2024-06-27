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

class ContinuousRobotNavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ContinuousRobotNavigationEnv, self).__init__()
        self.grid_size = 10
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # Continuous action space

        self.num_humans = 5  # Number of humans
        self.human_positions = [self._random_position() for _ in range(self.num_humans)]
        self.goal_position = (9, 9)

        self.reset()

    def _random_position(self):
        return [np.random.uniform(0, self.grid_size), np.random.uniform(0, self.grid_size)]

    def reset(self):
        self.robot_position = [0, 0]
        self.human_positions = [self._random_position() for _ in range(self.num_humans)]
        self.state = np.array(self.robot_position, dtype=np.float32)
        return self.state

    def step(self, action):
        # Scale action to the grid size
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.robot_position[0] += action[0]
        self.robot_position[1] += action[1]

        self.robot_position = np.clip(self.robot_position, 0, self.grid_size-1)
        self.state = np.array(self.robot_position, dtype=np.float32)

        done = np.array_equal(self.robot_position, list(self.goal_position))
        reward = 1 if done else -0.1

        # Update human positions
        for i in range(self.num_humans):
            human_action = np.random.uniform(-1, 1, 2)
            self.human_positions[i][0] += human_action[0]
            self.human_positions[i][1] += human_action[1]
            self.human_positions[i] = np.clip(self.human_positions[i], 0, self.grid_size-1)

        # Check for collisions with humans
        for human_pos in self.human_positions:
            if np.linalg.norm(np.array(self.robot_position) - np.array(human_pos)) < 1:
                reward = -1
                done = True

        truncated = False  # Add logic for truncation if needed

        return self.state, reward, done, truncated, {}

    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.goal_position] = 1
        grid[tuple(map(int, self.robot_position))] = 2
        for human_pos in self.human_positions:
            grid[tuple(map(int, human_pos))] = -1

        plt.imshow(grid, cmap='hot', interpolation='nearest')
        plt.show()

# Register the environment with OpenAI Gym
from gym.envs.registration import register

register(
    id='ContinuousRobotNavigation-v0',
    entry_point='__main__:ContinuousRobotNavigationEnv',
    max_episode_steps=100,
)


# if __name__ == '__main__':
#     env = gym.make('ContinuousRobotNavigation-v0')
#     state = env.reset()
#     for _ in range(20):
#         action = env.action_space.sample()
#         state, reward, done, truncated, _ = env.step(action)
#         env.render()
#         if done or truncated:
#             break

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4)

        self.max_action = max_action
        self.replay_buffer = []

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, iterations):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        for it in range(iterations):
            # Sample a batch of transitions from replay buffer
            batch = random.sample(self.replay_buffer, BATCH_SIZE)
            state, next_state, action, reward, done = zip(*batch)
            state = torch.FloatTensor(np.array(state)).to(device)
            next_state = torch.FloatTensor(np.array(next_state)).to(device)
            action = torch.FloatTensor(np.array(action)).to(device)
            reward = torch.FloatTensor(np.array(reward)).to(device)
            done = torch.FloatTensor(np.array(done)).to(device)

            # Compute the target Q value
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = reward + ((1 - done) * discount * torch.min(target_Q1, target_Q2)).detach()

            # Get current Q estimates
            current_Q1 = self.critic1(state, action)
            current_Q2 = self.critic2(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def add_to_replay_buffer(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))
        if len(self.replay_buffer) > REPLAY_BUFFER_SIZE:
            self.replay_buffer.pop(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discount = 0.99
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_delay = 2
REPLAY_BUFFER_SIZE = 1e6
BATCH_SIZE = 100

if __name__ == '__main__':
    env = gym.make('ContinuousRobotNavigation-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    td3 = TD3(state_dim, action_dim, max_action)

    # Training loop
    num_episodes = 100
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(100):
            action = td3.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            td3.add_to_replay_buffer(state, action, next_state, reward, done or truncated)
            state = next_state
            episode_reward += reward
            if done or truncated:
                break
        td3.train(100)
        episode_rewards.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward}")

    # Plot cumulative rewards
    plt.plot(range(num_episodes), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward over Episodes')
    plt.show()
