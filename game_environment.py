import pygame
import sys
import random
import numpy as np

pygame.init()

# Game Environment Interface for RL
class SnakeEnv:
    def __init__(self, screen_width=400, screen_height=400, block_size=20):
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.BLOCK_SIZE = block_size
        self.grid_width = self.SCREEN_WIDTH // self.BLOCK_SIZE
        self.grid_height = self.SCREEN_HEIGHT // self.BLOCK_SIZE

        self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font = pygame.font.SysFont(None, 30)
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake_alive = True
        self.snake_size = 3
        self.initial_x = 100
        self.initial_y = 100
        self.snake_body = [(self.initial_x, self.initial_y), (self.initial_x-self.BLOCK_SIZE, self.initial_y), (self.initial_x-(self.BLOCK_SIZE * 2), self.initial_y)]
        self.snake_dir = (1, 0)
        self.next_dir = self.snake_dir
        self.food = (100, 200)
        
        return self.get_observation()

    def get_observation(self):
        
        # 20x20x3, where its represnted as [is_snake_here, is_food_here, is_empty_here] and each value must be mutually exclusive with only one '1' per 3D array
        grid = np.zeros((self.SCREEN_WIDTH // self.BLOCK_SIZE, self.SCREEN_HEIGHT // self.BLOCK_SIZE, 3), dtype=np.float32)

        # index 0 is for snake body
        # 1 for true, 0 for false ^
        for (x, y) in self.snake_body:
            grid[y // self.BLOCK_SIZE][x // self.BLOCK_SIZE][0] = 1 

        # index 1 is for food
        fx, fy = self.food
        grid[fy // self.BLOCK_SIZE, fx // self.BLOCK_SIZE][1] = 1 

        # index 2 is for empty tile
        # 1 - food - snake, since either the tile is food or snake so if theres food then 1-1-0 = 0 so false
        grid[:, :, 2] = 1 - grid[:, :, 1] - grid[:, :, 0]

        return grid.flatten() # converts multi-D array into 1D

    def step(self, action):

        reward = -0.01 # small penalty per step to incentivize faster food seeking

        directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        proposed_dir = directions[action]

        # prevent going opposite direction in one move
        if (proposed_dir[0] * -1, proposed_dir[1] * -1) != self.snake_dir:
            self.snake_dir = proposed_dir

        (dx, dy) = self.snake_dir
        (x, y) = self.snake_body[0]
        new_head = (x + dx * self.BLOCK_SIZE, y + dy * self.BLOCK_SIZE)

        reward = 0
        done = False

        x, y = new_head
        if x < 0 or x >= self.SCREEN_WIDTH or y < 0 or y >= self.SCREEN_HEIGHT:
            self.snake_alive = False
            reward = -1
            done = True

        # check if new head collisions with body before insertion
        elif new_head in self.snake_body:
            self.snake_alive = False
            reward = -1
            done = True
        else:
            self.snake_body.insert(0, new_head)

            if new_head == self.food:
                self.snake_size += 1
                reward = 1
                self.randomize_food()
            else:
                if len(self.snake_body) > self.snake_size:
                    self.snake_body.pop()
                reward = 0

        obs = self.get_observation()

        return obs, reward, done
    
    def randomize_food(self):
        self.food = (random.randrange(0, self.SCREEN_WIDTH, self.BLOCK_SIZE), random.randrange(0, self.SCREEN_HEIGHT, self.BLOCK_SIZE))

        while self.food in self.snake_body:
            self.food = (random.randrange(0, self.SCREEN_WIDTH, self.BLOCK_SIZE), random.randrange(0, self.SCREEN_HEIGHT, self.BLOCK_SIZE))

    def draw_snake(self):
        for i, segment in enumerate(self.snake_body):

            max_green = 255
            min_green = 90 # not fully dark
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
            else:
                env.window.blit(env.font.render("Dead", True, 'red'), (env.SCREEN_WIDTH // 2 - 30, env.SCREEN_HEIGHT // 2))
                env.window.blit(env.font.render("Score: " + str(env.snake_size - 3), True, 'white'), (env.SCREEN_WIDTH // 2 - 30, env.SCREEN_HEIGHT // 2 + 30))

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        env.reset()
            
        pygame.display.flip()
        env.clock.tick(60)