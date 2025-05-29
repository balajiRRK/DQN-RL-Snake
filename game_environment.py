import pygame
import sys
import random
import numpy as np
import math

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
        self.randomize_food()
        
        return self.get_observation()

    def get_observation(self):
        head_x, head_y = self.snake_body[0]

        # Translate vector to direction string
        dir_vec_to_str = {
            (0, -1): 'UP',
            (0, 1): 'DOWN',
            (-1, 0): 'LEFT',
            (1, 0): 'RIGHT'
        }
        direction_str = dir_vec_to_str[self.snake_dir]

        # One-hot encoding for direction: [UP, DOWN, LEFT, RIGHT]
        direction = [
            1 if direction_str == 'UP' else 0,
            1 if direction_str == 'DOWN' else 0,
            1 if direction_str == 'LEFT' else 0,
            1 if direction_str == 'RIGHT' else 0,
        ]

        def is_danger(pos):
            x, y = pos
            if x < 0 or x >= self.SCREEN_WIDTH or y < 0 or y >= self.SCREEN_HEIGHT:
                return True
            return (x, y) in self.snake_body

        def move_in_direction(direction):
            if direction == 'UP':
                return (head_x, head_y - self.BLOCK_SIZE)
            elif direction == 'DOWN':
                return (head_x, head_y + self.BLOCK_SIZE)
            elif direction == 'LEFT':
                return (head_x - self.BLOCK_SIZE, head_y)
            elif direction == 'RIGHT':
                return (head_x + self.BLOCK_SIZE, head_y)

        # Mapping current direction to left/right/straight
        left_dir = {'UP': 'LEFT', 'DOWN': 'RIGHT', 'LEFT': 'DOWN', 'RIGHT': 'UP'}
        right_dir = {'UP': 'RIGHT', 'DOWN': 'LEFT', 'LEFT': 'UP', 'RIGHT': 'DOWN'}

        dir_straight = direction_str
        dir_left = left_dir[direction_str]
        dir_right = right_dir[direction_str]

        danger_straight = 1 if is_danger(move_in_direction(dir_straight)) else 0
        danger_left = 1 if is_danger(move_in_direction(dir_left)) else 0
        danger_right = 1 if is_danger(move_in_direction(dir_right)) else 0

        fx, fy = self.food
        food_up = 1 if fy < head_y else 0
        food_down = 1 if fy > head_y else 0
        food_left = 1 if fx < head_x else 0
        food_right = 1 if fx > head_x else 0

        observation = np.array(
            direction +
            [danger_straight, danger_left, danger_right] +
            [food_up, food_down, food_left, food_right],
            dtype=np.float32
        )
        
        return observation



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
        old_distance_to_food = math.sqrt((old_x - food_x) ** 2 + (old_y - food_y) ** 2)
        new_distance_to_food = math.sqrt((x - food_x) ** 2 + (y - food_y) ** 2)
        if old_distance_to_food > new_distance_to_food:
            reward += 0.1
        else:
            reward -= 0.05

        x, y = new_head
        if x < 0 or x >= self.SCREEN_WIDTH or y < 0 or y >= self.SCREEN_HEIGHT:
            self.snake_alive = False
            reward -= 1
            done = True

        # check if new head collisions with body before insertion
        elif new_head in self.snake_body:
            self.snake_alive = False
            reward -= 1
            done = True
        else:
            self.snake_body.insert(0, new_head)

            if new_head == self.food:
                self.snake_size += 1
                reward += 1
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
