import pygame
import sys
import random
import numpy as np

pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 400, 400
BLOCK_SIZE = 20

window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
window.fill(pygame.Color('black'))
font = pygame.font.SysFont(None, 30)
clock = pygame.time.Clock()

# snake initialization
initial_x = 100
initial_y = 100
snake_body = [(initial_x, initial_y), (initial_x-BLOCK_SIZE, initial_y), (initial_x-(BLOCK_SIZE * 2), initial_y)]
snake_dir = (1, 0) # (Horizontal (1 = right, -1 = left), Vertical (-1 = up, 1 = down)
snake_size = 3
snake_alive = True
next_dir = snake_dir

# food initialization
food = (100, 200)

def draw_snake(window):
    for i, segment in enumerate(snake_body):

        max_green = 255
        min_green = 90 # not fully dark
        green_fade = int(max_green - (i / len(snake_body)) * (max_green - min_green))
        color = (0, green_fade, 0)

        (x, y) = segment
        rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)

        pygame.draw.rect(window, color, rect)

def move_snake():
    global snake_size
    global snake_alive

    (dx, dy) = snake_dir
    (x, y) = snake_body[0] # head coords
    new_head = (x + (dx * BLOCK_SIZE), y + (dy * BLOCK_SIZE))
    

    # wall collision detection
    x, y = new_head
    if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
        snake_alive = False

    # check if new head collisions with body before insertion
    if new_head in snake_body:
        snake_alive = False

    snake_body.insert(0, new_head)

    # eats fruit
    if new_head == food:
        snake_size += 1
        randomize_food()

    if len(snake_body) > snake_size:
        snake_body.pop()

def draw_food(window):    
    x, y = food
    rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)

    pygame.draw.rect(window, (255, 0, 0), rect)

def randomize_food():
    global food
    food = (random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE), random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE))

    while food in snake_body:
        food = (random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE), random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE))

def reset():
    global snake_alive, snake_size, snake_body, snake_dir, next_dir

    snake_alive = True
    snake_size = 3
    snake_body = [(initial_x, initial_y), (initial_x-BLOCK_SIZE, initial_y), (initial_x-(BLOCK_SIZE * 2), initial_y)]
    snake_dir = (1, 0)
    next_dir = snake_dir

# Game Environment Interface for RL

def get_observation():
    
    # 20x20x3, where its represnted as [is_snake_here, is_food_here, is_empty_here] and each value must be mutually exclusive with only one '1' per 3D array
    grid = np.zeros((SCREEN_WIDTH // BLOCK_SIZE, SCREEN_HEIGHT // BLOCK_SIZE, 3), dtype=np.float32)

    # index 0 is for snake body
    # 1 for true, 0 for false ^
    for (x, y) in snake_body:
        grid[y // BLOCK_SIZE][x // BLOCK_SIZE][0] = 1 

    # index 1 is for food
    fx, fy = food
    grid[fy // BLOCK_SIZE, fx // BLOCK_SIZE][1] = 1 

    # index 2 is for empty tile
    # 1 - food - snake, since either the tile is food or snake so if theres food then 1-1-0 = 0 so false
    grid[:, :, 2] = 1 - grid[:, :, 1] - grid[:, :, 0]

    return grid.flatten() # converts multi-D array into 1D

def step(action):
    global snake_alive, snake_dir, snake_size

    directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    proposed_dir = directions[action]

    if (proposed_dir[0] * -1, proposed_dir[1] * -1) != snake_dir:
        snake_dir = proposed_dir

    (dx, dy) = snake_dir
    (x, y) = snake_body[0]
    new_head = (x + dx * BLOCK_SIZE, y + dy * BLOCK_SIZE)

    reward = 0
    done = False

    x, y = new_head
    if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
        snake_alive = False
        reward = -1
        done = True

    # check if new head collisions with body before insertion
    elif new_head in snake_body:
        snake_alive = False
        reward = -1
        done = True
    else:
        snake_body.insert(0, new_head)

        if new_head == food:
            snake_size += 1
            reward = 1
            randomize_food()
        else:
            if len(snake_body) > snake_size:
                snake_body.pop()
            reward = 0

    obs = get_observation()

    return obs, reward, done

# allows fps to be high while limiting game tickspeed
GAME_TICK = pygame.USEREVENT
pygame.time.set_timer(GAME_TICK, 150) # 150ms default

# Main Game Loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if snake_alive:

            # update dir not move snake on keypress since movement should be gated by game tickspeed while keypress handle should be handled constantly (at a 60fps rate)
            # updates next_dir and not snake_dir since user can queue multiple movement commands during 1 game tick which shouldn't be allowed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and snake_dir != (-1, 0):
                    next_dir = (1, 0)
                elif event.key == pygame.K_LEFT and snake_dir != (1, 0):
                    next_dir = (-1, 0)
                elif event.key == pygame.K_UP and snake_dir != (0, 1):
                    next_dir = (0, -1)
                elif event.key == pygame.K_DOWN and snake_dir != (0, -1):
                    next_dir = (0, 1)

            if event.type == GAME_TICK:
                snake_dir = next_dir
                window.fill('black')
                draw_snake(window)
                draw_food(window)
                move_snake()
        else:
            window.blit(font.render("Dead", True, 'red'), (SCREEN_WIDTH // 2 - 30, SCREEN_HEIGHT // 2))
            window.blit(font.render("Score: " + str(snake_size - 3), True, 'white'), (SCREEN_WIDTH // 2 - 30, SCREEN_HEIGHT // 2 + 30))

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset()
        
    pygame.display.flip()
    clock.tick(60)