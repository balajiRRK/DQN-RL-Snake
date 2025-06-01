import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
from collections import deque
from game_environment import SnakeEnv
import imageio
import matplotlib.pyplot as plt
import time

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.00
EPSILON_DECAY = 0.9992
TARGET_UPDATE_FREQ = 10
EPISODES = 5000

RENDER_EVERY = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

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
    
# Training loop
def train(model_path=None):

    start_time = time.time()

    env = SnakeEnv()
    obs_size = env.get_observation().shape[0]
    n_actions = 4 # right, left, up or down

    policy_net = DQN(obs_size, n_actions).to(device)
    target_net = DQN(obs_size, n_actions).to(device)

    if model_path is not None:
        print(f"Loading model from: {model_path}")
        policy_net.load_state_dict(torch.load(model_path, map_location=device))

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = 0 if model_path is not None else EPSILON_START # can replace this line with epsilon = EPSILON_START to continue training a pre-trained model rather than testing with eps = 0

    best_rendered_score = -float('inf')
    best_score = -float('inf')
    video_filename = 'best_snake_episode.mp4'
    model_save_path = "best_dqn_model.pth"
    all_scores = []
    all_losses = []

    for episode in range(EPISODES):
        obs = env.reset()
        total_reward = 0
        done = False

        render = (episode % RENDER_EVERY == 0)
        frames = []
        episode_losses = []

        while not done:

            if render:
                pygame.event.pump()
                env.window.fill('black')
                env.draw_snake()
                env.draw_food()
                env.display_score()
                pygame.display.flip()
                env.clock.tick(30)

                frame = pygame.surfarray.array3d(env.window)
                frame = frame.transpose([1, 0, 2]) # transpose (Width, Height, Channel) to (Height, Width, Channel)
                frames.append(frame) 

            # choose between random action and policy net action
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    action = policy_net(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).argmax().item()
            
            # Step through environment
            next_obs, reward, done = env.step(action)
            memory.push((obs, action, reward, next_obs, done))
            obs = next_obs
            total_reward += reward

            # Train the network
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

                obs_batch = torch.from_numpy(np.array(obs_batch)).float().to(device)
                action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1).to(device)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(device)
                next_obs_batch = torch.from_numpy(np.array(next_obs_batch)).float().to(device)
                done_batch = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1).to(device)

                #1 Forward pass
                q_values = policy_net(obs_batch).gather(1, action_batch)
                with torch.no_grad():
                    max_next_q = target_net(next_obs_batch).max(1)[0].unsqueeze(1)
                    target_q = reward_batch + GAMMA * max_next_q * (1 - done_batch)

                #2 Calculate loss
                loss = nn.MSELoss()(q_values, target_q)

                #3 Optimizer zero grad zeros out gradients since gradients can accumulate into next iteration even though they have already been applied 
                optimizer.zero_grad()

                #4 Backprop
                loss.backward()

                #5 Gradient Descent
                optimizer.step()

                # visualization
                episode_losses.append(loss.item())
            
        score = env.snake_size - env.STARTING_SIZE

        # visualization
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        all_losses.append(avg_loss)
        all_scores.append(score)

        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        print(f"Episode {episode}, Total reward: {round(total_reward, 2)}, Score: {score}, Epsilon: {epsilon:.3f}")

        # NOTE: only saves best score episode AMONG the few ones recorded...
        if render and score > best_rendered_score:
            best_rendered_score = score
            print(f"New best score {best_rendered_score} at episode {episode}. Saving video...")
            imageio.mimwrite(video_filename, frames, fps=15, codec='libx264', quality=8)
        
        if score > best_score: # best total score among all episodes
            best_score = score
            print(f"New best model score: {best_score}. Saving model weights...")
            torch.save(policy_net.state_dict(), model_save_path)

    
    # After training complete:
    window = 50
    episodes = np.arange(len(all_scores))
    rolling_scores = np.convolve(all_scores, np.ones(window)/window, mode='valid')
    rolling_losses = np.convolve(all_losses, np.ones(window)/window, mode='valid')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(episodes[len(episodes) - len(rolling_scores):], rolling_scores, label=f"Avg Score ({window}-episode window)")
    ax1.set_title("Smoothed Score per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.grid(True)

    ax2.plot(episodes[len(episodes) - len(rolling_losses):], rolling_losses, label=f"Avg Loss ({window}-episode window)", color="orange")
    ax2.set_title("Smoothed Loss per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png")

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\nTotal training time: {minutes} minutes and {seconds} seconds")

    print(f"\nBest score from a single episode in training: {best_score}")
        
if __name__ == "__main__":
    try:
        model_path = sys.argv[1] if len(sys.argv) > 1 else None # assign model path if given or set to None if not
        train(model_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        pygame.quit()