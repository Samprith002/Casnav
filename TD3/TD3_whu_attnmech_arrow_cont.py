import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
from matplotlib.patches import FancyArrow
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContinuousRobotNavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def is_goal_behind_wall(self, goal_position):
        for wall in self.maze_walls:
            x, y, width, height = wall
            if goal_position[0] >= x and goal_position[0] <= x + width and goal_position[1] >= y and goal_position[1] <= y + height:
                return True
        return False
    

    def __init__(self):
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
    
        return self.state, reward, done, {}


    def render(self, mode='human'):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
    
        # Set the background color
        ax.set_facecolor('lightgray')
        fig.patch.set_facecolor('lightgray')
    
        # Plot the goal position
        goal = plt.Circle(self.goal_position, 0.5, color='darkgreen')
        ax.add_artist(goal)
    
        # Plot the robot position
        robot = plt.Circle(self.robot_position, 0.5, color='darkblue')
        ax.add_artist(robot)

    # Plot human positions
        for human_pos in self.human_positions:
            human = plt.Circle(human_pos, 0.5, color='darkred')
            ax.add_artist(human)
    
        # Plot the maze walls
        for wall in self.maze_walls:
            x, y, width, height = wall
            rect = plt.Rectangle((x, y), width, height, color='black')
            ax.add_patch(rect)
    
        # Plot observation range, ensuring it doesn't pass through walls
        from shapely.geometry import Point, LineString, box
    
        def intersects_wall(start, end):
            line = LineString([start, end])
            for wall in self.maze_walls:
                wall_box = box(wall[0], wall[1], wall[0] + wall[2], wall[1] + wall[3])
                if line.intersects(wall_box):
                    return True
            return False
    
        observation_range = self.observation_radius
        angles = np.linspace(0, 2 * np.pi, 360)
        x0, y0 = self.robot_position

        for angle in angles:
            x1 = x0 + observation_range * np.cos(angle)
            y1 = y0 + observation_range * np.sin(angle)
            if not intersects_wall((x0, y0), (x1, y1)):
                line = plt.Line2D((x0, x1), (y0, y1), color='gray', alpha=0.2)
                ax.add_line(line)
    
        # Plot predicted human trajectories only if within observation range
        for predicted_positions in self.predicted_human_positions:
            for future_pos in predicted_positions:
                if np.linalg.norm(np.array(future_pos) - np.array(self.robot_position)) <= self.observation_radius:
                    if not intersects_wall(self.robot_position, future_pos):
                        future_circle = plt.Circle(future_pos, 0.4, color='indigo', alpha=0.3)
                        ax.add_artist(future_circle)
    
        # Plot the direction arrow of the robot
        arrow_length = 0.7
        dx = arrow_length * np.cos(self.robot_orientation)
        dy = arrow_length * np.sin(self.robot_orientation)
        arrow = FancyArrow(self.robot_position[0], self.robot_position[1], dx, dy, color='black', width=0.1)
        ax.add_patch(arrow)
    
        # Hide the axis
        plt.axis('off')
    
        # Save the frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frame = imageio.imread(buf)
        self.frames.append(frame)
        buf.close()



from gym.envs.registration import register

register(
    id='ContinuousRobotNavigation-v0',
    entry_point='__main__:ContinuousRobotNavigationEnv',
    max_episode_steps=100,
)

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)  # Project back to the original dimension
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        queries = self.query_layer(x)
        keys = self.key_layer(x)
        values = self.value_layer(x)
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(queries.size(-1))
        attention_weights = self.softmax(scores)
        
        attended_values = torch.matmul(attention_weights, values)
        attended_values = self.output_layer(attended_values)  # Project back to the original dimension
        return attended_values, attention_weights


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.attention = Attention(state_dim, 128)
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        attended_state, _ = self.attention(state)
        
        a = F.relu(self.l1(attended_state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.attention = Attention(state_dim, 128)
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        attended_state, _ = self.attention(state)
        sa = torch.cat([attended_state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2



class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
        )


class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1
        
        # Sample a batch of transitions from replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state))[0].mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


discount = 0.99
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_delay = 2
REPLAY_BUFFER_SIZE = 1e6
BATCH_SIZE = 100

def main():
    env = ContinuousRobotNavigationEnv()
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]  # Flatten the observation space
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    td3_agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Training loop
    num_episodes = 100
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(100):
            action = td3_agent.select_action(state.flatten())
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state.flatten(), action, next_state.flatten(), reward, done)
            state = next_state
            episode_reward += reward
            env.render()  # Render the environment at each step
            if done:
                break
        td3_agent.train(replay_buffer, batch_size=100)
        episode_rewards.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward}")

        # Save GIF
        imageio.mimsave(f'episode_{episode}.gif', env.frames, fps=10)

    plt.plot(range(num_episodes), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward over Episodes')
    plt.show()

if __name__ == "__main__":
    main()
