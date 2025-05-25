import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from game_environment import SnakeEnv

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
# Replay buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
# Trainig loop
def train():
    env = SnakeEnv()
    obs_size = env.get_observation().shape[0]
    n_actions = 4 # right, left, up down, 4 total options

    policy_net = DQN(obs_size, n_actions)
    target_net = DQN(obs_size, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPSILON_START

    for episode in range(1000):
        obs = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    action = policy_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax().item()
            
            next_obs, reward, done = env.step(action)
            memory.push((obs, action, reward, next_obs, done))
            obs = next_obs
            total_reward += reward

            # Train the network
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

                obs_batch = torch.tensor(obs_batch, dtype=torch.float32)
                action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)
                next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32)
                done_batch = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1)

                q_values = policy_net(obs_batch).gather(1, action_batch)
                with torch.no_grad():
                    max_next_q = target_net(next_obs_batch).max(1)[0].unsqueeze(1)
                    target_q = reward_batch + GAMMA * max_next_q * (1 - done_batch)

                loss = nn.MSELoss()(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon:.3f}")
        
if __name__ == "__main__":
    train()