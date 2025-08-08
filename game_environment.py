import pygame
import sys
import random
import numpy as np

pygame.init()

# Game Environment Interface for RL
class SnakeEnv:
    def __init__(self, screen_width=400, screen_height=400, block_size=20, starting_size=3):
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.BLOCK_SIZE = block_size
        self.STARTING_SIZE = starting_size

        self.grid_width = self.SCREEN_WIDTH // self.BLOCK_SIZE
        self.grid_height = self.SCREEN_HEIGHT // self.BLOCK_SIZE

        self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font = pygame.font.SysFont('Arial', 15, bold=False)
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake_alive = True
        self.snake_size = self.STARTING_SIZE
        self.initial_x = 100
        self.initial_y = 100
        self.snake_body = [(self.initial_x, self.initial_y), (self.initial_x-self.BLOCK_SIZE, self.initial_y), (self.initial_x-(self.BLOCK_SIZE * 2), self.initial_y)]
        self.snake_dir = (1, 0)
        self.next_dir = self.snake_dir
        self.randomize_food()

        return self.get_observation()

    # creating obs by using 4 separate channels (2D arrays) for each info type instead of one 2D array with different numbers corresponding to different
    # info types since that would require the NN to learn the differences on its own
    def get_observation(self):
        grid_h, grid_w = self.grid_height, self.grid_width

        # Create 4 channels: body, head, food, direction
        body_channel = np.zeros((grid_h, grid_w), dtype=np.float32)
        head_channel = np.zeros((grid_h, grid_w), dtype=np.float32)
        food_channel = np.zeros((grid_h, grid_w), dtype=np.float32)
        direction_channel = np.zeros((grid_h, grid_w), dtype=np.float32)

        # Body (excluding head)
        for (x, y) in self.snake_body[1:]:
            gx = x // self.BLOCK_SIZE
            gy = y // self.BLOCK_SIZE
            body_channel[gy, gx] = 1.0

        # Head
        head_x, head_y = self.snake_body[0]
        gx = head_x // self.BLOCK_SIZE
        gy = head_y // self.BLOCK_SIZE
        head_channel[gy, gx] = 1.0

        # Food
        fx, fy = self.food
        gx = fx // self.BLOCK_SIZE
        gy = fy // self.BLOCK_SIZE
        food_channel[gy, gx] = 1.0

        # Direction (encoded only at head position)
        if self.snake_dir == (0, -1): # up
            direction_channel[gy, gx] = 0.25
        elif self.snake_dir == (0, 1): # down
            direction_channel[gy, gx] = 0.5
        elif self.snake_dir == (-1, 0): # left
            direction_channel[gy, gx] = 0.75
        elif self.snake_dir == (1, 0): # right
            direction_channel[gy, gx] = 1.0

        # Stack channels into a single observation tensor
        obs = np.stack([body_channel, head_channel, food_channel, direction_channel], axis=0)  # shape: (4, H, W)
        return obs

    def step(self, action):

        reward = -0.01 # small penalty per step to incentivize faster food seeking

        directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        proposed_dir = directions[action]

        # prevent going opposite direction in one move
        if (proposed_dir[0] * -1, proposed_dir[1] * -1) != self.snake_dir:
            self.snake_dir = proposed_dir

        (dx, dy) = self.snake_dir
        old_x, old_y = self.snake_body[0]
        new_head = (old_x + dx * self.BLOCK_SIZE, old_y + dy * self.BLOCK_SIZE)

        done = False
        x, y = new_head

        # incentivize moving snake closer to food
        food_x, food_y = self.food
        old_distance_to_food = abs(old_x - food_x) + abs(old_y - food_y)
        new_distance_to_food = abs(x - food_x) + abs(y - food_y)
        if old_distance_to_food > new_distance_to_food:
            reward += 0.1
        else:
            reward -= 0.1

        x, y = new_head
        if x < 0 or x >= self.SCREEN_WIDTH or y < 0 or y >= self.SCREEN_HEIGHT:
            self.snake_alive = False
            reward -= 5
            done = True

        # check if new head collisions with body before insertion
        elif new_head in self.snake_body:
            self.snake_alive = False
            reward -= 5
            done = True
        else:
            self.snake_body.insert(0, new_head)

            if new_head == self.food:
                self.snake_size += 1
                reward += 15
                self.randomize_food()
            else:
                if len(self.snake_body) > self.snake_size:
                    self.snake_body.pop()

        obs = self.get_observation()

        return obs, reward, done

    def randomize_food(self):
        self.food = (random.randrange(0, self.SCREEN_WIDTH, self.BLOCK_SIZE), random.randrange(0, self.SCREEN_HEIGHT, self.BLOCK_SIZE))

        while self.food in self.snake_body:
            self.food = (random.randrange(0, self.SCREEN_WIDTH, self.BLOCK_SIZE), random.randrange(0, self.SCREEN_HEIGHT, self.BLOCK_SIZE))

    def draw_snake(self):
        for i, segment in enumerate(self.snake_body):

            max_green = 255
            min_green = 90 # not fully dark so doesnt blend in with black background
            green_fade = int(max_green - (i / len(self.snake_body)) * (max_green - min_green))
            color = (0, green_fade, 0)

            (x, y) = segment
            rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)

            pygame.draw.rect(self.window, color, rect)

    def draw_food(self):
        x, y = self.food
        rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        pygame.draw.rect(self.window, (255, 0, 0), rect)

    def move_snake(self):

        (dx, dy) = self.snake_dir
        (x, y) = self.snake_body[0] # head coords
        new_head = (x + (dx * self.BLOCK_SIZE), y + (dy * self.BLOCK_SIZE))

        # wall collision detection
        x, y = new_head
        if x < 0 or x >= self.SCREEN_WIDTH or y < 0 or y >= self.SCREEN_HEIGHT:
            self.snake_alive = False

        # check if new head collisions with body before insertion
        if new_head in self.snake_body:
            self.snake_alive = False

        self.snake_body.insert(0, new_head)

        # eats fruit
        if new_head == self.food:
            self.snake_size += 1
            self.randomize_food()
 
        if len(self.snake_body) > self.snake_size:
            self.snake_body.pop()

    def display_score(self):
        self.window.blit(self.font.render("Score: " + str(self.snake_size - self.STARTING_SIZE), True, 'white'), (0, 0))

# ------- MANUAL Snake -------

# only execute if game_env file was executed, not if the file is imported
if __name__ == "__main__": 
        
    env = SnakeEnv()

    # allows fps to be high while limiting game tickspeed
    GAME_TICK = pygame.USEREVENT
    pygame.time.set_timer(GAME_TICK, 150) # 150ms default

    # Main Game Loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if env.snake_alive:

                # update dir not move snake on keypress since movement should be gated by game tickspeed while keypress handle should be handled constantly (at a 60fps rate)
                # updates next_dir and not snake_dir since user can queue multiple movement commands during 1 game tick which shouldn't be allowed
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT and env.snake_dir != (-1, 0):
                        env.next_dir = (1, 0)
                    elif event.key == pygame.K_LEFT and env.snake_dir != (1, 0):
                        env.next_dir = (-1, 0)
                    elif event.key == pygame.K_UP and env.snake_dir != (0, 1):
                        env.next_dir = (0, -1)
                    elif event.key == pygame.K_DOWN and env.snake_dir != (0, -1):
                        env.next_dir = (0, 1)

                if event.type == GAME_TICK:
                    env.snake_dir = env.next_dir
                    env.window.fill('black')
                    env.draw_snake()
                    env.draw_food()
                    env.move_snake()
                    env.display_score()
            else:
                env.window.blit(env.font.render("Dead", True, 'red'), (env.SCREEN_WIDTH // 2 - 30, env.SCREEN_HEIGHT // 2))

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        env.reset()
            
        pygame.display.flip()
        env.clock.tick(60)
