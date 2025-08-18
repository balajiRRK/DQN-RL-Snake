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
LR = 1e-4
BATCH_SIZE = 32
MEMORY_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.01
TARGET_UPDATE_FREQ = 100
EPISODES = 1000
        
LOG_INTERVAL = 100
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
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1),  # -> (16, grid_h, grid_w)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),                            # -> (32, grid_h, grid_w)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),                            # -> (32, grid_h, grid_w)
            nn.ReLU()
        )
        
        # AFTER these three conv layers (with padding=1), the spatial dimensions remain (grid_h × grid_w).
        # Therefore the flattened size is:
        conv_output_size = 32 * grid_h * grid_w

        self.fc = nn.Sequential(
            nn.Flatten(),                      # Flattens (batch, 32, grid_h, grid_w) -> (batch, 32*grid_h*grid_w)
            nn.Linear(conv_output_size, 256),  # Now uses the *actual* conv_output_size, not a hard-coded 25600
            nn.ReLU(),
            nn.Linear(256, n_actions)          # One Q-value per action
        )

    def forward(self, x):
        # Expect x of shape (batch, 4, grid_h, grid_w)
        x = self.conv(x)   # -> (batch, 64, grid_h, grid_w)
        return self.fc(x)  # -> (batch, n_actions)

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.positive_buffer = deque(maxlen=15000) # Store good experiences longer
        
    def push(self, transition):
        state, action, reward, next_state, done = transition
        self.buffer.append(transition)
        
        # Keep positive rewards in separate buffer
        if reward > 3:  # Food reward is +10 and lets say -0.5 from 50 steps and -5 from death so 4.5, so this catches food-eating experiences
            self.positive_buffer.append(transition)
    
    def sample(self, batch_size):
        # If we have positive experiences, include more of them in the batch
        if len(self.positive_buffer) > 0:
            # Sample 60% from regular buffer, 40% from positive buffer
            regular_size = int(batch_size * 0.6)
            positive_size = min(batch_size - regular_size, len(self.positive_buffer))
            remaining_size = batch_size - positive_size
            
            regular_sample = random.sample(self.buffer, min(remaining_size, len(self.buffer)))
            positive_sample = random.sample(self.positive_buffer, positive_size)
            
            return regular_sample + positive_sample
        else:
            # Fall back to regular sampling if no positive experiences yet
            return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)
    
def train(model_path=None, epsilon_decay=0.99975):

    start_time = time.time()

    env = SnakeEnv()
    grid_h, grid_w = env.grid_height, env.grid_width
    n_actions = 4  # right, left, up, down

    policy_net = DQN(grid_h, grid_w, n_actions).to(device)
    target_net = DQN(grid_h, grid_w, n_actions).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    best_rendered_score = -float("inf")
    best_score = -float("inf")
    video_filename = "best_snake_episode.mp4"
    model_save_path = "best_model.pth"
    checkpoint_save_path = "training_checkpoint.pth"

    all_scores = []
    all_losses = []

    if model_path is not None:
        print(f"Loading model from file: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False) # Don't run with untrusted files

        if isinstance(checkpoint, dict) and "policy_net_state" in checkpoint:
            print("Model weights and additional training details found from file: {model_path}")
            policy_net.load_state_dict(checkpoint["policy_net_state"])
            target_net.load_state_dict(checkpoint["target_net_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            epsilon = 0.2
            EPSILON_DECAY = 0.998
            
            all_scores = checkpoint.get("scores", [])
            all_losses = checkpoint.get("losses", [])
            resume_training_points = checkpoint.get("resume_training_points", [0])
            start_episode = len(all_scores)

            resume_training_points.append(start_episode)
        else:
            print("Only model weights found from file: {model_path}")
            policy_net.load_state_dict(checkpoint)
            target_net.load_state_dict(checkpoint)
            optimizer = optim.Adam(policy_net.parameters(), lr=LR)
            all_scores, all_losses = [], []
            resume_training_points = [0]
            start_episode = 0
            epsilon = 0.2
            EPSILON_DECAY = 0.998
    else:
        start_episode = 0
        resume_training_points = [0]
        epsilon = EPSILON_START

    try: 
        for episode in range(start_episode, EPISODES + start_episode + 1):
            obs = env.reset()
            total_reward = 0
            steps_since_last_fruit = 0
            done = False

            render = episode % RENDER_EVERY == 0
            frames = []
            episode_losses = []

            while not done:

                if render:
                    pygame.event.pump()
                    env.window.fill("black")
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
                        # Select best actions using policy_net
                        best_next_actions = policy_net(next_obs_batch).argmax(1).unsqueeze(1)  # (batch, 1)
                        
                        # Evaluate those actions using target_net
                        target_q_values = target_net(next_obs_batch).gather(1, best_next_actions)
                        
                        target_q = reward_batch + GAMMA * target_q_values * (1 - done_batch)

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
                        print(f"Episode {episode} ended due to performing {steps_since_last_fruit} steps without touching food")
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

            if episode % LOG_INTERVAL == 0 and episode != 0:
                recent_scores = all_scores[-LOG_INTERVAL:] if len(all_scores) >= LOG_INTERVAL else all_scores
                avg_recent_score = np.mean(recent_scores)
                print(f"Average score from episodes {episode - LOG_INTERVAL}-{episode}: {avg_recent_score:.2f}")

            # NOTE: only saves best score episode AMONG the few ones recorded...
            if render and score > best_rendered_score:
                best_rendered_score = score
                print(f"New best video score {best_rendered_score} at episode {episode}. Saving video...")
                imageio.mimwrite(video_filename, frames, fps=15, codec='libx264', quality=8)
            
            if score > best_score: # best total score among all episodes
                best_score = score
                print(f"New best overall score: {best_score}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user\n")
    finally:
        torch.save(policy_net.state_dict(), model_save_path) # save just the model for testing purposes
        torch.save({
            "policy_net_state": policy_net.state_dict(),
            "target_net_state": target_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scores": all_scores,
            "losses": all_losses,
            "resume_training_points": resume_training_points,

        }, checkpoint_save_path) # save model and other info to continue training

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

        # after you plot curves on ax1 and ax2:
        if len(resume_training_points) > 1:
            # label just the first resume line so the legend isn't spammy
            first = True
            for i, ep in enumerate(resume_training_points[1:], start=1):  # skip the initial 0
                label = "Continued Training Start" if first else None
                ax1.axvline(x=ep, color='red', linestyle='--', linewidth=2, label=label)
                ax2.axvline(x=ep, color='red', linestyle='--', linewidth=2, label=label if first else None)
                first = False

                # Optional: add tiny vertical labels near the top of the axes (axis-relative y=0.95)
                ax1.text(ep, 0.95, f"Resume #{i}", rotation=90, va='top', ha='right',
                        color='red', transform=ax1.get_xaxis_transform())
                ax2.text(ep, 0.95, f"Resume #{i}", rotation=90, va='top', ha='right',
                        color='red', transform=ax2.get_xaxis_transform())


        ax2.set_title("Smoothed Loss per Episode")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.grid(True)

        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        
        training_info = f"Training Time: {minutes}m {seconds}s | Episodes: {episode} | Best Score: {best_score} | Epsilon Decay: {EPSILON_DECAY}"
        fig.suptitle(f"Snake AI Training Results - {training_info}", fontsize=12, y=0.98)   

        ax1.legend()
        ax2.legend()

        plt.tight_layout()
        plt.savefig("training_metrics.png")

        print(f"\nTotal training time: {minutes} minutes and {seconds} seconds")

        print(f"\nBest score from a single episode in training: {best_score}")

def test(model_path, episodes=50, render_every=1):
    env = SnakeEnv()
    grid_h, grid_w = env.grid_height, env.grid_width
    n_actions = 4  # right, left, up, down
    
    policy_net = DQN(grid_h, grid_w, n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    video_filename = 'best_snake_episode.mp4'
    best_rendered_score = -float('inf')

    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        
        frames = []
        render = episode % render_every == 0
 
        while not done:
            pygame.event.pump()
            env.window.fill('black')
            env.draw_snake()
            env.draw_food()
            env.display_score()
            pygame.display.flip()
            env.clock.tick(15)

            # Choose best action without exploration
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy_net(obs_tensor).argmax().item()
            
            obs, reward, done = env.step(action)
            total_reward += reward

            if render:
                frame = pygame.surfarray.array3d(env.window)
                frame = frame.transpose([1, 0, 2]) # transpose (Width, Height, Channel) to (Height, Width, Channel)
                frames.append(frame)

        score = env.snake_size - env.STARTING_SIZE
        print(f"Episode {episode}, Total reward: {round(total_reward, 2)}, Score: {score}")

        # NOTE: only saves best score episode AMONG the (potentially) few ones recorded...
        if render and score > best_rendered_score:
            best_rendered_score = score
            print(f"New best score {best_rendered_score} at episode {episode}. Saving video...")
            imageio.mimwrite(video_filename, frames, fps=15, codec='libx264', quality=8)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python script.py <train|test> <model_weights_filename>.pth")
        sys.exit(1)

    mode = sys.argv[1]
    model_path_arg = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        if mode == "train":
            train(model_path_arg)
        elif mode == "test":
            if model_path_arg is None:
                print("Error: Please provide a model path for testing")
                sys.exit(1)
            test(model_path_arg)
        else:
            print("Error: Invalid mode, use either \"train\" or \"test\"")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        pygame.quit()