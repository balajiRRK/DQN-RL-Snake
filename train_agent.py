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
EPSILON_END = 0.05
EPSILON_DECAY = 0.999
TARGET_UPDATE_FREQ = 10
EPISODES = 3000

RENDER_EVERY = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# CNN
class DQN(nn.Module):
    def __init__(self, grid_h, grid_w, n_actions):
        super().__init__()
        
        # Our observation is now 4 channels (body, head, food, direction).
        # So Conv2d must expect in_channels=4.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),  # -> (32, grid_h, grid_w)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),                            # -> (64, grid_h, grid_w)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),                            # -> (64, grid_h, grid_w)
            nn.ReLU()
        )
        
        # AFTER these three conv layers (with padding=1), the spatial dimensions remain (grid_h × grid_w).
        # Therefore the flattened size is:
        conv_output_size = 64 * grid_h * grid_w

        self.fc = nn.Sequential(
            nn.Flatten(),                      # Flattens (batch, 64, grid_h, grid_w) -> (batch, 64*grid_h*grid_w)
            nn.Linear(conv_output_size, 512),  # Now uses the *actual* conv_output_size, not a hard-coded 25600
            nn.ReLU(),
            nn.Linear(512, n_actions)          # One Q-value per action
        )

    def forward(self, x):
        # Expect x of shape (batch, 4, grid_h, grid_w)
        x = self.conv(x)   # -> (batch, 64, grid_h, grid_w)
        return self.fc(x)  # -> (batch, n_actions)

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        # transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
def train(model_path=None):

    start_time = time.time()

    env = SnakeEnv()
    grid_h, grid_w = env.grid_height, env.grid_width
    n_actions = 4  # right, left, up, down

    policy_net = DQN(grid_h, grid_w, n_actions).to(device)
    target_net = DQN(grid_h, grid_w, n_actions).to(device)

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
        steps_since_last_fruit = 0
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
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action = policy_net(obs_tensor).argmax().item()
            
            # Step through environment
            next_obs, reward, done = env.step(action)
            memory.push((obs, action, reward, next_obs, done))
            obs = next_obs
            total_reward += reward

            # Train the network
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

                # Convert to tensors:
                obs_batch = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=device)
                action_batch = torch.tensor(action_batch, dtype=torch.int64, device=device).unsqueeze(1)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device).unsqueeze(1)
                next_obs_batch = torch.tensor(np.stack(next_obs_batch), dtype=torch.float32, device=device)
                done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device).unsqueeze(1)

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

                if reward > 0:
                    steps_since_last_fruit = 0
                else:
                    steps_since_last_fruit += 1

                if steps_since_last_fruit > 100:
                    print(f"Episode {episode} ended due to performing {steps_since_last_fruit} without touching food.")
                    done = True
            
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