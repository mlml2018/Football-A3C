
import pygame
import numpy as np
import math

# freepik.com
# flaticon.com

# Initialise pygame
pygame.init()

# Create the screen
screen_width = 1024
screen_height = 720
#screen_width = 800
#screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Scale based on screen size
base_width = 1280
base_height = 1024
height_scale = screen_height / base_height

# Player and Ball
# player_width = int(25 * height_scale)
# player_height = player_width
# ball_width = int(10 * height_scale)
# ball_height = ball_width
player_width = 23
player_height = 23
keeper_width = 18
keeper_height = 18
ball_width = 8
ball_height = 8

import os
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')

ballImg = pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_DIR, 'football.png')), (ball_width, ball_height))
playerImg = pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_DIR, 'player.png')), (player_width, player_height))
defenderImg = pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_DIR, 'defender.png')), (player_width, player_height))
keeperImg = pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_DIR, 'goalkeeper.png')),
                                   (keeper_width, keeper_height))

def player(x, y):
    screen.blit(playerImg, (x, y))

def keeper(x, y):
    screen.blit(keeperImg, (x, y))

def draw_players(player_loc):
    for i in range(len(player_loc[0])-3):
        #draw player whether they are facing left or right
        if player_loc[2][i] == 0:
            screen.blit(playerImg, (player_loc[0][i], player_loc[1][i]))
        elif player_loc[2][i] == 1:
            screen.blit(pygame.transform.flip(playerImg, True, False),
                        (player_loc[0][i], player_loc[1][i]))
    screen.blit(defenderImg, (player_loc[0][2], player_loc[1][2]))
    screen.blit(ballImg, (player_loc[0][-1], player_loc[1][-1]))
    keeper(player_loc[0][-2], player_loc[1][-2])




# Player list (initial location)
player_loc = np.array([[200*height_scale, 1000*height_scale, 628*height_scale, 628*height_scale, 222*height_scale],
                       [800*height_scale, 800*height_scale, 180*height_scale, 80*height_scale, 820*height_scale],
                       [0, 0, 0, 0, 0],
                       [False, False, False, False, False]])
# Attacker 1, Attacker 2, Defender, Keeper, Ball
# x coordinate, y coordinate, player orientation, player that passed/shot the ball

player_vel = int(3)
defender_vel = int(2)
keeper_vel = int(1)


def player1_movement(keys_pressed, player_loc):
    possess_ball = math.hypot(player_loc[0][0]+10-player_loc[0][-1],player_loc[1][0]+25-player_loc[1][-1]) < 20
    dx, dy = 0, 0

    # Horizontal movement
    if keys_pressed[pygame.K_a]:
        dx -= 1
    if keys_pressed[pygame.K_d]:
        dx += 1

    # Vertical movement
    if keys_pressed[pygame.K_w]:
        dy -= 1
    if keys_pressed[pygame.K_s]:
        dy += 1

    # Normalize velocity for diagonal movement
    if dx != 0 and dy != 0:
        scale = 1 / np.sqrt(2)  # Scale by sqrt(2) to maintain constant velocity
        dx *= scale
        dy *= scale
    
    # Update player location
    if possess_ball:
        player_loc[0][0] += dx * player_vel*3/5
        player_loc[1][0] += dy * player_vel*3/5
    else:
        player_loc[0][0] += dx * player_vel
        player_loc[1][0] += dy * player_vel
    
    if player_loc[0][0] <= 115*height_scale - player_width/2:
        player_loc[0][0] = 115*height_scale - player_width/2
    if player_loc[0][0] >= 1165*height_scale - player_width/2:
        player_loc[0][0] = 1165*height_scale - player_width/2
    if player_loc[1][0] <= 56*height_scale - player_height/2:
        player_loc[1][0] = 56*height_scale - player_height/2
    if player_loc[1][0] >= 861*height_scale - player_height/2:
        player_loc[1][0] = 861*height_scale - player_height/2

    # if player_loc[0][0] <= 0:
    #     player_loc[0][0] = 0
    # if player_loc[0][0] >= screen_width - player_width:
    #     player_loc[0][0] = screen_width - player_width
    # if player_loc[1][0] <= int(36/25*player_height):
    #     player_loc[1][0] = int(36/25*player_height)
    # if player_loc[1][0] >= int(861/25*player_height):
    #     player_loc[1][0] = int(861/25*player_height)
    # if player_loc[1][0] <= 36:
    #     player_loc[1][0] = 36
    # if player_loc[1][0] >= 861:
    #     player_loc[1][0] = 861



def player2_movement(keys_pressed,player_loc):
    possess_ball = math.hypot(player_loc[0][1]+10-player_loc[0][-1],player_loc[1][1]+25-player_loc[1][-1]) < 20
    dx, dy = 0, 0

    # Horizontal movement
    if keys_pressed[pygame.K_LEFT]:
        dx -= 1
    if keys_pressed[pygame.K_RIGHT]:
        dx += 1

    # Vertical movement
    if keys_pressed[pygame.K_UP]:
        dy -= 1
    if keys_pressed[pygame.K_DOWN]:
        dy += 1

    # Normalize velocity for diagonal movement
    if dx != 0 and dy != 0:
        scale = 1 / np.sqrt(2)  # Scale by sqrt(2) to maintain constant velocity
        dx *= scale
        dy *= scale

    # Update player location
    if possess_ball:
        player_loc[0][1] += dx * player_vel*3/5
        player_loc[1][1] += dy * player_vel*3/5
    else:
        player_loc[0][1] += dx * player_vel
        player_loc[1][1] += dy * player_vel


    if player_loc[0][1] <= 115*height_scale - player_width/2:
        player_loc[0][1] = 115*height_scale - player_width/2
    if player_loc[0][1] >= 1165*height_scale - player_width/2:
        player_loc[0][1] = 1165*height_scale - player_width/2
    if player_loc[1][1] <= 56*height_scale - player_height/2:
        player_loc[1][1] = 56*height_scale - player_height/2
    if player_loc[1][1] >= 861*height_scale - player_height/2:
        player_loc[1][1] = 861*height_scale - player_height/2

def defender_movement(player_loc):
    # Distance between ball and defender
    dx = (player_loc[0][-1]-player_loc[0][2])
    dy = (player_loc[1][-1]-player_loc[1][2])
    d = math.hypot(dx,dy)

    player_loc[0][2] += dx/d * defender_vel
    player_loc[1][2] += dy/d * defender_vel


    if player_loc[0][2] <= 115*height_scale - player_width/2:
        player_loc[0][2] = 115*height_scale - player_width/2
    if player_loc[0][2] >= 1165*height_scale - player_width/2:
        player_loc[0][2] = 1165*height_scale - player_width/2
    if player_loc[1][2] <= 56*height_scale - player_height/2:
        player_loc[1][2] = 56*height_scale - player_height/2
    if player_loc[1][2] >= 861*height_scale - player_height/2:
        player_loc[1][2] = 861*height_scale - player_height/2

def keeper_movement(keys_pressed, player_loc):

    if keys_pressed[pygame.K_z]:
        player_loc[0][-2] -= keeper_vel
    if keys_pressed[pygame.K_x]:
        player_loc[0][-2] += keeper_vel
    if keys_pressed[pygame.K_c]:
        player_loc[1][-2] -= keeper_vel
    if keys_pressed[pygame.K_v]:
        player_loc[1][-2] += keeper_vel

    if player_loc[0][-2] <= 307*height_scale+keeper_width:
        player_loc[0][-2] = 307*height_scale+keeper_width
    if player_loc[0][-2] >= 923*height_scale:
        player_loc[0][-2] = 923*height_scale
    if player_loc[1][-2] <= 40:
        player_loc[1][-2] = 40
    if player_loc[1][-2] >= 283*height_scale:
        player_loc[1][-2] = 283*height_scale


# Pass and Shoot
pass_and_shoot = np.array([[0, 0, 0, 0, False], [0, 0, 0, 0, False]])
# mx,my,x speed,y speed,activation
shooting_vel = 7
passing_vel = 5.5

def pass_shoot(mx, my, keys_pressed, player_loc, pass_and_shoot):

    # check distance between player and ball
    for i in range(len(player_loc[0])-3):
        # if math.hypot(player_loc[0][i]+20-player_loc[0][-1],
        #               player_loc[1][i]+50-player_loc[1][-1]) < 50:
        if math.hypot(player_loc[0][i]+10-player_loc[0][-1],player_loc[1][i]+25-player_loc[1][-1]) < 20:
            dist = math.hypot(mx-player_loc[0][-1], my-player_loc[1][-1])
            if keys_pressed[pygame.K_SPACE]:  # pass
                pass_and_shoot[0][0] = mx
                pass_and_shoot[0][1] = my
                pass_and_shoot[0][2] = (mx-player_loc[0][-1])/dist*passing_vel
                pass_and_shoot[0][3] = (my-player_loc[1][-1])/dist*passing_vel
                pass_and_shoot[0][-1] = True
                player_loc[3][i] = True

            elif keys_pressed[pygame.K_LSHIFT]:  # shoot
                pass_and_shoot[1][0] = mx
                pass_and_shoot[1][1] = my
                pass_and_shoot[1][2] = (mx-player_loc[0][-1])/dist*shooting_vel
                pass_and_shoot[1][3] = (my-player_loc[1][-1])/dist*shooting_vel
                pass_and_shoot[1][-1] = True
                player_loc[3][i] = True

    if pass_and_shoot[0][-1]:
        player_loc[0][-1] += pass_and_shoot[0][2]
        player_loc[1][-1] += pass_and_shoot[0][3]

    if pass_and_shoot[1][-1]:
        player_loc[0][-1] += pass_and_shoot[1][2]
        player_loc[1][-1] += pass_and_shoot[1][3]

# Out of bounds
def out_of_bounds(player_loc, pass_and_shoot):

    if player_loc[0][-1] <= 104*height_scale or player_loc[0][-1] >= 1166*height_scale or player_loc[1][-1] <= 45*height_scale or player_loc[1][-1] >= 862*height_scale:
        player_loc[0][-1] = 222 * height_scale
        player_loc[1][-1] = 820 * height_scale
        player_loc[0][0] = 200 * height_scale
        player_loc[0][1] = 1000 * height_scale
        player_loc[1][0] = 800 * height_scale
        player_loc[1][1] = 800 * height_scale
        pass_and_shoot[0][-1] = False
        pass_and_shoot[1][-1] = False
        for i in range(len(player_loc)-2):
            player_loc[3][i] = False

# Possession of ball
def possess_ball(player_loc, pass_and_shoot):

    for i in range(len(player_loc[0])-2):
        if math.hypot(player_loc[0][i]+10-player_loc[0][-1],
                      player_loc[1][i]+25-player_loc[1][-1]) < 20 and not player_loc[3][i]:
            pass_and_shoot[0][-1] = False
            pass_and_shoot[1][-1] = False
            for j in range(len(player_loc)-2):
                player_loc[3][j] = False
            if player_loc[2][i] == 0:
                player_loc[0][-1] = player_loc[0][i] + 20
            elif player_loc[2][i] == 1:
                player_loc[0][-1] = player_loc[0][i] - 5

            player_loc[1][-1] = player_loc[1][i] + 25

    # goalkeeper makes a save / defender steals ball
    if math.hypot(player_loc[0][-2]+keeper_width/2-player_loc[0][-1]-ball_width/2,
                  player_loc[1][-2]+keeper_height/2-player_loc[1][-1]-ball_height/2) < 13 or \
                  math.hypot(player_loc[0][2]+keeper_width/2-player_loc[0][-1]-ball_width/2,
                  player_loc[1][2]+keeper_height/2-player_loc[1][-1]-ball_height/2) < 20:
        pass_and_shoot[0][-1] = False
        pass_and_shoot[1][-1] = False
        for j in range(len(player_loc)-2):
            player_loc[3][j] = True


# Goal
goal_bool = False
def goal(player_loc, pass_and_shoot):
    global goal_bool
    if player_loc[1][-1] <= 50*height_scale and player_loc[0][-1] >= 584*height_scale and player_loc[0][-1] <= 696*height_scale:
        goal_bool = True

    if goal_bool and player_loc[1][-1] <= 30:
        pass_and_shoot[0][-1] = False
        pass_and_shoot[1][-1] = False

# Goal
goalImg = pygame.image.load(os.path.join(ASSETS_DIR, 'goal-box-with-net.png'))

# Draw Windows
def draw_window():
    screen.fill((0, 128, 0))
    screen.blit(pygame.transform.scale(goalImg, (112*height_scale, 60*height_scale)), (584*height_scale, 0))

    # https://en.wikipedia.org/wiki/Football_pitch#/media/File:Soccer_pitch_dimensions.png
    # scale factor 14, 115 - 75 , need to change scale factor
    
    # side lines
    pygame.draw.line(screen, (255, 255, 255), (115*height_scale, 56*height_scale), (1165*height_scale, 56*height_scale), 1)
    pygame.draw.line(screen, (255, 255, 255), (115*height_scale, 56*height_scale), (115*height_scale, 861*height_scale), 1)
    pygame.draw.line(screen, (255, 255, 255), (115*height_scale, 861*height_scale), (1165*height_scale, 861*height_scale), 1)
    pygame.draw.line(screen, (255, 255, 255), (1165*height_scale, 861*height_scale), (1165*height_scale, 56*height_scale), 1)

    # penalty area
    pygame.draw.line(screen, (255, 255, 255), (500*height_scale, 56*height_scale), (500*height_scale, 146*height_scale), 1)
    pygame.draw.line(screen, (255, 255, 255), (500*height_scale, 146*height_scale), (780*height_scale, 146*height_scale), 1)
    pygame.draw.line(screen, (255, 255, 255), (780*height_scale, 146*height_scale), (780*height_scale, 56*height_scale), 1)

    pygame.draw.line(screen, (255, 255, 255), (332*height_scale, 56*height_scale), (332*height_scale, 308*height_scale), 1)
    pygame.draw.line(screen, (255, 255, 255), (332*height_scale, 308*height_scale), (948*height_scale, 308*height_scale), 1)
    pygame.draw.line(screen, (255, 255, 255), (948*height_scale, 308*height_scale), (948*height_scale, 56*height_scale), 1)

    pygame.draw.circle(screen, (255, 255, 255), (640*height_scale, 224*height_scale), 4)
    pygame.draw.arc(screen, (255, 255, 255), [500*height_scale, 84*height_scale, 280*height_scale, 280*height_scale], np.pi + np.arcsin(0.6), np.pi*2 - np.arcsin(0.6))

    # Centre circle
    pygame.draw.arc(screen, (255, 255, 255), [500*height_scale, 721*height_scale, 280*height_scale, 280*height_scale], 0, np.pi, 1)
    pygame.draw.circle(screen, (255, 255, 255), (640*height_scale, 861*height_scale), 7*height_scale)


# Game Loop

FPS = 60

def main():
    clock = pygame.time.Clock()
    running = True
    mx, my = 400, 20
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                running = False

        keys_pressed = pygame.key.get_pressed()
        possess_ball(player_loc, pass_and_shoot)
        player1_movement(keys_pressed, player_loc)
        player2_movement(keys_pressed, player_loc)
        defender_movement(player_loc)
        keeper_movement(keys_pressed, player_loc)
        pass_shoot(mx, my, keys_pressed, player_loc, pass_and_shoot)
        goal(player_loc, pass_and_shoot)
        if not goal_bool:
            out_of_bounds(player_loc, pass_and_shoot)

        draw_window()
        draw_players(player_loc)
        pygame.display.update()

if __name__ == "__main__":

    main()
