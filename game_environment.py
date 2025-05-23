import pygame
import sys
import random

pygame.init()

screen_width, screen_height = 400, 400
window = pygame.display.set_mode((screen_width, screen_height))
window.fill(pygame.Color('black'))
font = pygame.font.SysFont(None, 30)
BLOCK_SIZE = 20
clock = pygame.time.Clock()

# snake initialization
initial_x = 100
initial_y = 100
snake_body = [(initial_x, initial_y), (initial_x-BLOCK_SIZE, initial_y), (initial_x-(BLOCK_SIZE * 2), initial_y)]
snake_dir = (1, 0) # (Horizontal (1 = right, -1 = left), Vertical (-1 = up, 1 = down)
SNAKE_SIZE = 3
snake_alive = True

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
    global SNAKE_SIZE
    global snake_alive

    (dx, dy) = snake_dir
    (x, y) = snake_body[0] # head coords
    new_head = (x + (dx * 20), y + (dy * 20))
    
    snake_body.insert(0, new_head)

    # eats fruit
    if new_head == food:
        SNAKE_SIZE += 1
        randomize_food()

    # collision detection
    x, y = new_head
    if x < 0 or x >= screen_width or y < 0 or y >= screen_height:
        snake_alive = False

    if new_head in snake_body[1:]:
        snake_alive = False

    if len(snake_body) > SNAKE_SIZE:
        snake_body.pop()

def draw_food(window):    
    x, y = food
    rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)

    pygame.draw.rect(window, (255, 0, 0), rect)

def randomize_food():
    global food
    food = (random.randrange(1, screen_width), random.randrange(1, screen_height))

    while food not in snake_body:
        food = (random.randrange(1, screen_width), random.randrange(1, screen_height))

# allows fps to be high while limiting game tickspeed
GAME_TICK = pygame.USEREVENT
pygame.time.set_timer(GAME_TICK, 100) 

# Main Game Loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if snake_alive:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and snake_dir != (-1, 0):
                    snake_dir = (1, 0) # update dir not move snake on keypress since movement should be gated by game tickspeed while keypress handle should be handled constantly (at a 60fps rate)
                elif event.key == pygame.K_LEFT and snake_dir != (1, 0):
                    snake_dir = (-1, 0)
                elif event.key == pygame.K_UP and snake_dir != (0, 1):
                    snake_dir = (0, -1)
                elif event.key == pygame.K_DOWN and snake_dir != (0, -1):
                    snake_dir = (0, 1)

            if event.type == GAME_TICK:
                window.fill('black')
                draw_snake(window)
                draw_food(window)
                move_snake()
        else:
            window.blit(font.render("Dead", True, 'red'), (screen_width // 2 - 30, screen_height // 2))
            window.blit(font.render("Score: " + str(SNAKE_SIZE - 3), True, 'white'), (screen_width // 2 - 30, screen_height // 2 + 30))
        
    pygame.display.flip()
    clock.tick(60)