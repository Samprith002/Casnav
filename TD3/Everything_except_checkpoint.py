import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from shapely.geometry import Point, LineString, box
import matplotlib.pyplot as plt
import io
import imageio
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import deque
import os
import gym
from gym import spaces
import numpy as np
import torch.nn.functional as F
import random
import copy
import warnings
from torch_geometric.nn import GATConv
import math
from matplotlib.patches import FancyArrow
from collections import namedtuple
from torch.distributions import Gumbel
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=True, dropout=0.6)
        self.fc_out = nn.Linear(out_channels * heads, out_channels)

    def forward(self, x, edge_index):
        x = self.gat_conv(x, edge_index)
        x = F.elu(x)
        x = self.fc_out(x)
        return x

class InteractionBasedAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(InteractionBasedAttentionModel, self).__init__()
        self.gat_layers = nn.ModuleList([
            GAT(in_channels=input_dim if i == 0 else hidden_dim, out_channels=hidden_dim, heads=num_heads)
            for i in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.gumbel_softmax = GumbelSoftmax()

    def forward(self, x, edge_index):
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
        x = self.fc_out(x)
        return self.gumbel_softmax(x)

    def predict(self, agent_positions, agent_velocities):
        x = torch.tensor(agent_positions + agent_velocities, dtype=torch.float32).to(device)
        edge_index = self.construct_edge_index(len(agent_positions) + len(agent_velocities))
        predicted_trajectories = self.forward(x, edge_index)
        return predicted_trajectories.detach().cpu().numpy()

    def construct_edge_index(self, num_agents):
        edge_index = []
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

class GumbelSoftmax(nn.Module):
    def __init__(self, temperature=1.0):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / self.temperature, dim=-1)

    def forward(self, logits):
        return self.gumbel_softmax_sample(logits)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class EnhancedGumbelSocialTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim, max_len=100):
        super(EnhancedGumbelSocialTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.gumbel_softmax = GumbelSoftmax()

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return self.gumbel_softmax(x)

    def predict(self, agent_positions, agent_velocities):
        input_data = torch.tensor(agent_positions + agent_velocities, dtype=torch.float32).unsqueeze(0).to(device)
        predicted_trajectories = self.forward(input_data)
        return predicted_trajectories.squeeze(0).detach().cpu().numpy()

import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import imageio
import torch
from matplotlib.patches import FancyArrow

class ContinuousRobotNavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, interaction_model):
        super(ContinuousRobotNavigationEnv, self).__init__()
        self.grid_size = 10
        self.num_obstacles = 3
        self.observation_radius = 4 
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.observation_radius * 2 + 1, self.observation_radius * 2 + 1), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # Continuous action space

        self.num_humans = 3  
        self.human_positions = [self._random_position() for _ in range(self.num_humans)]
        self.human_velocities = [self._random_velocity() for _ in range(self.num_humans)]
        self.obstacle_positions = [self._random_position() for _ in range(self.num_obstacles)]
        self.lidar_range = 4.0
        self.goal_position = (9, 9)
        self.predicted_human_positions = [[] for _ in range(self.num_humans)]

        self.maze_walls = [
            (5, 8, 0.3, 1.5),  
            (0, 7, 1.5, 0.3),  
            (4, 4.5, 0.3, 1.5),  
            (5, 2.5, 1.5, 0.3), 
            (8, 4, 1.5, 0.3)  
        ]

        self.frames = []  
        self.interaction_model = interaction_model

        self.reset()

    def _random_position(self):
        return [np.random.uniform(0, self.grid_size - 1), np.random.uniform(0, self.grid_size - 1)]

    def _random_velocity(self):
        return [np.random.uniform(-0.25, 0.25), np.random.uniform(-0.25, 0.25)]

    def reset(self):
        self.robot_orientation = np.pi/4  
        self.robot_position = [0, 0]
        self.human_positions = [self._random_position() for _ in range(self.num_humans)]
        self.human_velocities = [self._random_velocity() for _ in range(self.num_humans)]
        self.obstacle_positions = [self._random_position() for _ in range(self.num_obstacles)]
        self.predicted_human_positions = [[] for _ in range(self.num_humans)]
        self.state = self._get_observation()
        while True:
            self.goal_position = self._random_position()
            if not self.is_goal_behind_wall(self.goal_position):
                break
        self.frames = [] 
        return self.state

    def is_goal_behind_wall(self, goal_position):
        for wall in self.maze_walls:
            x, y, width, height = wall
            if goal_position[0] >= x and goal_position[0] <= x + width and goal_position[1] >= y and goal_position[1] <= y + height:
                return True
        return False

    def _get_observation(self):
        obs = np.zeros((self.observation_radius * 2 + 1, self.observation_radius * 2 + 1))

        for i in range(-self.observation_radius, self.observation_radius + 1):
            for j in range(-self.observation_radius, self.observation_radius + 1):
                xi = int(self.robot_position[0] + i)
                yj = int(self.robot_position[1] + j)
                if 0 <= xi < self.grid_size and 0 <= yj < self.grid_size:
                    if (xi, yj) == self.goal_position:
                        obs[i + self.observation_radius, j + self.observation_radius] = 1
                    elif any(np.linalg.norm(np.array([xi, yj]) - np.array(human_pos)) < 1.0 for human_pos in self.human_positions):
                        obs[i + self.observation_radius, j + self.observation_radius] = -1
                    elif [xi, yj] == self.robot_position:
                        obs[i + self.observation_radius, j + self.observation_radius] = 2

        return obs

    def prepare_trajectory_data(self):
        agent_positions = self.human_positions + [self.robot_position]
        agent_velocities = self.human_velocities + [[0, 0]]  # Robot velocity can be assumed as [0, 0] initially
        return agent_positions, agent_velocities

    def predict_human_positions(self):
        for i in range(self.num_humans):
            next_position = [
                self.human_positions[i][0] + self.human_velocities[i][0],
                self.human_positions[i][1] + self.human_velocities[i][1]
            ]
    
            if next_position[0] < 0 or next_position[0] >= self.grid_size:
                self.human_velocities[i][0] *= -1
            if next_position[1] < 0 or next_position[1] >= self.grid_size:
                self.human_velocities[i][1] *= -1
    
            for wall in self.maze_walls:
                x, y, width, height = wall
                if (x <= next_position[0] <= x + width) and (y <= next_position[1] <= y + height):
                    if x <= next_position[0] <= x + width:
                        self.human_velocities[i][0] *= -1
                    if y <= next_position[1] <= y + height:
                        self.human_velocities[i][1] *= -1
    
            self.human_positions[i][0] += self.human_velocities[i][0]
            self.human_positions[i][1] += self.human_velocities[i][1]
    
            future_positions = []
            future_position = self.human_positions[i].copy()
            future_velocity = self.human_velocities[i].copy()
            for _ in range(5): 
                future_position[0] += future_velocity[0]
                future_position[1] += future_velocity[1]
    
                if future_position[0] < 0 or future_position[0] >= self.grid_size:
                    future_velocity[0] *= -1
                if future_position[1] < 0 or future_position[1] >= self.grid_size:
                    future_velocity[1] *= -1
    
                for wall in self.maze_walls:
                    x, y, width, height = wall
                    if (x <= future_position[0] <= x + width) and (y <= future_position[1] <= y + height):
                        if x <= future_position[0] <= x + width:
                            future_velocity[0] *= -1
                        if y <= future_position[1] <= y + height:
                            future_velocity[1] *= -1
    
                future_positions.append(future_position.copy())
    
            self.predicted_human_positions[i] = future_positions

    def check_collision(self):
        for human_pos in self.human_positions:
            if np.linalg.norm(np.array(self.robot_position) - np.array(human_pos)) < 1.0:  # Collision threshold
                return True
    
        for obstacle_pos in self.obstacle_positions:
            if np.linalg.norm(np.array(self.robot_position) - np.array(obstacle_pos)) < 1.0:  # Collision threshold
                return True
    
        for wall in self.maze_walls:
            x, y, width, height = wall
            if (x <= self.robot_position[0] <= x + width) and (y <= self.robot_position[1] <= y + height):
                return True

        return False

    def step(self, action):
        self.previous_position = self.robot_position.copy()
        self.robot_position[0] += action[0]
        self.robot_position[1] += action[1]
    
        self.robot_position[0] = np.clip(self.robot_position[0], 0, self.grid_size - 1)
        self.robot_position[1] = np.clip(self.robot_position[1], 0, self.grid_size - 1)
    
        if np.linalg.norm(action) > 0:
            self.robot_orientation = np.arctan2(action[1], action[0])
    
        agent_positions, agent_velocities = self.prepare_trajectory_data()
        predicted_trajectories = self.interaction_model.predict(agent_positions, agent_velocities)
    
        # Use the predicted trajectories to modify the robot's actions
        for traj in predicted_trajectories:
            if np.linalg.norm(np.array(self.robot_position) - np.array(traj[:2])) < self.observation_radius:
                # Assuming traj has the shape [x, y]
                self.robot_position[0] -= traj[0] * 0.1
                self.robot_position[1] -= traj[1] * 0.1
    
        self.predict_human_positions()

        distance_to_goal = np.linalg.norm(np.array(self.robot_position) - np.array(self.goal_position))
        reward = -distance_to_goal
    
        if distance_to_goal < 0.5:
            reward += 100
            done = True
        else:
            collision = self.check_collision()
            if collision:
                reward -= 10
                done = True
            else:
                done = False
    
        self.capture_frame()
        return self.state, reward, done, {}


    def render(self, mode='human'):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
    
        ax.set_facecolor('lightgray')
        fig.patch.set_facecolor('lightgray')
    
        goal = plt.Circle(self.goal_position, 0.5, color='darkgreen')
        ax.add_artist(goal)
    
        robot = plt.Circle(self.robot_position, 0.5, color='darkblue')
        ax.add_artist(robot)
        
        arrow_length = 0.5
        dx = arrow_length * np.cos(self.robot_orientation)
        dy = arrow_length * np.sin(self.robot_orientation)
        arrow = FancyArrow(self.robot_position[0], self.robot_position[1], dx, dy, head_width=0.3, head_length=0.3, fc='yellow', ec='yellow')
        ax.add_patch(arrow)
    
        for i, human_pos in enumerate(self.human_positions):
            human = plt.Circle(human_pos, 0.5, color='darkred')
            ax.add_artist(human)
            if np.linalg.norm(np.array(self.robot_position) - np.array(human_pos)) <= self.observation_radius:
                predicted_positions = self.predicted_human_positions[i]
                for pos in predicted_positions:
                    predicted_circle = plt.Circle(pos, 0.2, color='red', fill=True)
                    ax.add_artist(predicted_circle)
    
        for wall in self.maze_walls:
            x, y, width, height = wall
            rect = plt.Rectangle((x, y), width, height, color='black')
            ax.add_patch(rect)
    
        observation_circle = plt.Circle(self.robot_position, self.observation_radius, color='blue', fill=False, linestyle='--')
        ax.add_artist(observation_circle)
    
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.frames.append(imageio.imread(buf))
        plt.close()

    def capture_frame(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.set_facecolor('lightgray')
        fig.patch.set_facecolor('lightgray')
        goal = plt.Circle(self.goal_position, 0.5, color='darkgreen')
        ax.add_artist(goal)
        robot = plt.Circle(self.robot_position, 0.5, color='darkblue')
        ax.add_artist(robot)
        arrow_length = 0.5
        dx = arrow_length * np.cos(self.robot_orientation)
        dy = arrow_length * np.sin(self.robot_orientation)
        arrow = FancyArrow(self.robot_position[0], self.robot_position[1], dx, dy, head_width=0.3, head_length=0.3, fc='yellow', ec='yellow')
        ax.add_patch(arrow)
        for i, human_pos in enumerate(self.human_positions):
            human = plt.Circle(human_pos, 0.5, color='darkred')
            ax.add_artist(human)
            if np.linalg.norm(np.array(self.robot_position) - np.array(human_pos)) <= self.observation_radius:
                predicted_positions = self.predicted_human_positions[i]
                for pos in predicted_positions:
                    predicted_circle = plt.Circle(pos, 0.2, color='red', fill=True)
                    ax.add_artist(predicted_circle)
        for wall in self.maze_walls:
            x, y, width, height = wall
            rect = plt.Rectangle((x, y), width, height, color='black')
            ax.add_patch(rect)
        observation_circle = plt.Circle(self.robot_position, self.observation_radius, color='blue', fill=False, linestyle='--')
        ax.add_artist(observation_circle)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.frames.append(imageio.imread(buf))
        plt.close()


from gym.envs.registration import register

register(
    id='ContinuousRobotNavigation-v0',
    entry_point='__main__:ContinuousRobotNavigationEnv',
    max_episode_steps=100,
)

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
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def add(self, state, action, next_state, reward, done):
        max_priority = max(self.priorities, default=1.0)
        experience = (state, action, next_state, reward, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        next_states = np.array(batch[2])
        rewards = np.array(batch[3])
        dones = np.array(batch[4])

        return states, actions, next_states, rewards, dones, weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

class TD3:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

        # For tracking losses
        self.actor_losses = []
        self.critic_losses = []

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, beta=0.4):
        self.total_it += 1

        # Sample replay buffer with PER
        state, action, next_state, reward, done, weights, indices = replay_buffer.sample(batch_size, beta)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)
        weights = torch.FloatTensor(weights).to(device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) * weights + F.mse_loss(current_Q2, target_Q) * weights
        prios = critic_loss + 1e-5
        critic_loss = critic_loss.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        replay_buffer.update_priorities(indices, prios.cpu().data.numpy())

        self.critic_losses.append(critic_loss.item())

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.actor_losses.append(actor_loss.item())

discount = 0.99
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_delay = 2
REPLAY_BUFFER_SIZE = 1e6
BATCH_SIZE = 100

def main():
    interaction_model = InteractionBasedAttentionModel(input_dim=2, hidden_dim=128, output_dim=2, num_heads=4, num_layers=6)
    interaction_model.to(device)
    env = ContinuousRobotNavigationEnv(interaction_model)
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    td3_agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = PrioritizedReplayBuffer(capacity=100000)
    
    num_episodes = 100
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(100):
            action = td3_agent.select_action(state.flatten())

            agent_positions, agent_velocities = env.prepare_trajectory_data()
            predicted_trajectories = interaction_model.predict(agent_positions, agent_velocities)

            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state.flatten(), action, next_state.flatten(), reward, done)
            state = next_state
            episode_reward += reward
            if done:
                break
    
        td3_agent.train(replay_buffer, batch_size=256, beta=0.4)
        episode_rewards.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward}")

        if env.frames:
            imageio.mimsave(f'episode_{episode}.gif', env.frames, fps=10)

    plt.plot(range(num_episodes), episode_rewards, label="Episode Rewards")
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward over Episodes')
    plt.legend()
    plt.show()

    plt.plot(td3_agent.actor_losses, label="Actor Losses")
    plt.plot(td3_agent.critic_losses, label="Critic Losses")
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.title('Actor and Critic Losses')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
