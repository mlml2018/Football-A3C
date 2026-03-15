import pygame
import numpy as np

pygame.init()
#font = pygame.font.Font('arial.ttf', 25)


# Player and Ball Sizes
player_width = 23
player_height = 23
keeper_width = 18
keeper_height = 18
ball_width = 8
ball_height = 8

# Player, Ball, Goal Post Images
ballImg = pygame.transform.scale(pygame.image.load('football.png'), (ball_width, ball_height))
playerImg = pygame.transform.scale(pygame.image.load('player.png'), (player_width, player_height))
keeperImg = pygame.transform.scale(pygame.image.load('goalkeeper.png'),(keeper_width, keeper_height))
defenderImg = pygame.transform.scale(pygame.image.load('defender.png'), (player_width, player_height))
goalImg = pygame.image.load('goal-box-with-net.png')

# Player vel
player_vel = int(3)
defender_vel = int(2)
keeper_vel = int(1)

# Ball vel
shooting_vel = 7
passing_vel = 5.5

FPS = 60


class Football_Game():

    def __init__(self, w=1280, h=1024):
        self.w = w
        self.h = h
        self.scale = self.h /1024

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Football')
        self.clock = pygame.time.Clock()
        self.reset()

    
    def reset(self):
        # init game state
        # x coordinate, y coordinate, player that passed ball
        self.player_ball_loc = np.array([[200*self.scale, 1000*self.scale, 628*self.scale, 628*self.scale, 222*self.scale],
                       [800*self.scale, 800*self.scale, 180*self.scale, 80*self.scale, 820*self.scale],
                       [False, False, False, False, False]])
        
        # mx (mouse x coordinate), my, speed x, speed y, activation
        # speed x, speed y, ball passed boolean
        self.pass_info = np.array([0, 0, False])
        self.mx, self.my = 1000*self.scale, 800*self.scale
        
    def _update_ui(self):
        self.display.fill((0, 128, 0))
        self.display.blit(pygame.transform.scale(goalImg, (112*self.scale, 60*self.scale)), (584*self.scale, 0))

        # side lines
        pygame.draw.line(self.display, (255, 255, 255), (115*self.scale, 56*self.scale), (1165*self.scale, 56*self.scale), 1)
        pygame.draw.line(self.display, (255, 255, 255), (115*self.scale, 56*self.scale), (115*self.scale, 861*self.scale), 1)
        pygame.draw.line(self.display, (255, 255, 255), (115*self.scale, 861*self.scale), (1165*self.scale, 861*self.scale), 1)
        pygame.draw.line(self.display, (255, 255, 255), (1165*self.scale, 861*self.scale), (1165*self.scale, 56*self.scale), 1)

        # penalty area
        pygame.draw.line(self.display, (255, 255, 255), (500*self.scale, 56*self.scale), (500*self.scale, 146*self.scale), 1)
        pygame.draw.line(self.display, (255, 255, 255), (500*self.scale, 146*self.scale), (780*self.scale, 146*self.scale), 1)
        pygame.draw.line(self.display, (255, 255, 255), (780*self.scale, 146*self.scale), (780*self.scale, 56*self.scale), 1)

        pygame.draw.line(self.display, (255, 255, 255), (332*self.scale, 56*self.scale), (332*self.scale, 308*self.scale), 1)
        pygame.draw.line(self.display, (255, 255, 255), (332*self.scale, 308*self.scale), (948*self.scale, 308*self.scale), 1)
        pygame.draw.line(self.display, (255, 255, 255), (948*self.scale, 308*self.scale), (948*self.scale, 56*self.scale), 1)

        pygame.draw.circle(self.display, (255, 255, 255), (640*self.scale, 224*self.scale), 4)
        pygame.draw.arc(self.display, (255, 255, 255), [500*self.scale, 84*self.scale, 280*self.scale, 280*self.scale], np.pi + np.arcsin(0.6), np.pi*2 - np.arcsin(0.6))

        # Centre circle
        pygame.draw.arc(self.display, (255, 255, 255), [500*self.scale, 721*self.scale, 280*self.scale, 280*self.scale], 0, np.pi, 1)
        pygame.draw.circle(self.display, (255, 255, 255), (640*self.scale, 861*self.scale), 7*self.scale)

        # Draw players and ball
        for i in range(len(self.player_ball_loc[0])-3):
            self.display.blit(playerImg, (self.player_ball_loc[0][i], self.player_ball_loc[1][i]))
        self.display.blit(ballImg, (self.player_ball_loc[0][-1], self.player_ball_loc[1][-1]))
        self.display.blit(defenderImg, (self.player_ball_loc[0][2], self.player_ball_loc[1][2]))
        self.display.blit(keeperImg, (self.player_ball_loc[0][-2], self.player_ball_loc[1][-2]))

        pygame.display.flip()
    
    def possess_ball(self):
        for i in range(len(self.player_ball_loc[0])-3):
            if np.hypot(self.player_ball_loc[0][i]+10-self.player_ball_loc[0][-1],
                        self.player_ball_loc[1][i]+25-self.player_ball_loc[1][-1]) < 20 and not self.player_ball_loc[-1][i]:
                self.pass_info[-1] = False
                self.player_ball_loc[-1][0] = False
                self.player_ball_loc[-1][1] = False
                self.player_ball_loc[0][-1] = self.player_ball_loc[0][i] + 20
                self.player_ball_loc[1][-1] = self.player_ball_loc[1][i] + 25
                

    def pass_shoot(self, keys_pressed):
        for i in range(len(self.player_ball_loc[0])-3):
            if np.hypot(self.player_ball_loc[0][i]+10-self.player_ball_loc[0][-1],self.player_ball_loc[1][i]+25-self.player_ball_loc[1][-1]) < 20:    
                dist = np.hypot(self.mx-self.player_ball_loc[0][-1], self.my-self.player_ball_loc[1][-1])
                if keys_pressed[pygame.K_SPACE] and dist > 0:  # pass
                    self.pass_info[0] = (self.mx-self.player_ball_loc[0][-1])/dist*passing_vel
                    self.pass_info[1] = (self.my-self.player_ball_loc[1][-1])/dist*passing_vel
                    self.player_ball_loc[-1][i] = True
                    self.pass_info[-1] = True

        if self.pass_info[-1]:
            self.player_ball_loc[0][-1] += self.pass_info[0]
            self.player_ball_loc[1][-1] += self.pass_info[1]

    def player1_movement(self, keys_pressed):
        possess_ball = np.hypot(self.player_ball_loc[0][0]+10-self.player_ball_loc[0][-1],self.player_ball_loc[1][0]+25-self.player_ball_loc[1][-1]) < 20
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
            self.player_ball_loc[0][0] += dx * player_vel*3/5
            self.player_ball_loc[1][0] += dy * player_vel*3/5
        
        else:
            self.player_ball_loc[0][0] += dx * player_vel
            self.player_ball_loc[1][0] += dy * player_vel
        
        if self.player_ball_loc[0][0] <= 115*self.scale - player_width/2:
            self.player_ball_loc[0][0] = 115*self.scale - player_width/2
        if self.player_ball_loc[0][0] >= 1165*self.scale - player_width/2:
            self.player_ball_loc[0][0] = 1165*self.scale - player_width/2
        if self.player_ball_loc[1][0] <= 56*self.scale - player_height/2:
            self.player_ball_loc[1][0] = 56*self.scale - player_height/2
        if self.player_ball_loc[1][0] >= 861*self.scale - player_height/2:
            self.player_ball_loc[1][0] = 861*self.scale - player_height/2
    
    def player2_movement(self, keys_pressed):
        possess_ball = np.hypot(self.player_ball_loc[0][1]+10-self.player_ball_loc[0][-1],self.player_ball_loc[1][1]+25-self.player_ball_loc[1][-1]) < 20
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
            self.player_ball_loc[0][1] += dx * player_vel*3/5
            self.player_ball_loc[1][1] += dy * player_vel*3/5

        else:
            self.player_ball_loc[0][1] += dx * player_vel
            self.player_ball_loc[1][1] += dy * player_vel


        if self.player_ball_loc[0][1] <= 115*self.scale - player_width/2:
            self.player_ball_loc[0][1] = 115*self.scale - player_width/2
        if self.player_ball_loc[0][1] >= 1165*self.scale - player_width/2:
            self.player_ball_loc[0][1] = 1165*self.scale - player_width/2
        if self.player_ball_loc[1][1] <= 56*self.scale - player_height/2:
            self.player_ball_loc[1][1] = 56*self.scale - player_height/2
        if self.player_ball_loc[1][1] >= 861*self.scale - player_height/2:
            self.player_ball_loc[1][1] = 861*self.scale - player_height/2

    def defender_movement(self):
        # Distance between ball and defender
        dx = (self.player_ball_loc[0][-1]-self.player_ball_loc[0][2])
        dy = (self.player_ball_loc[1][-1]-self.player_ball_loc[1][2])
        d = np.hypot(dx,dy)

        self.player_ball_loc[0][2] += dx/d * defender_vel
        self.player_ball_loc[1][2] += dy/d * defender_vel


        if self.player_ball_loc[0][2] <= 115*self.scale - player_width/2:
            self.player_ball_loc[0][2] = 115*self.scale - player_width/2
        if self.player_ball_loc[0][2] >= 1165*self.scale - player_width/2:
            self.player_ball_loc[0][2] = 1165*self.scale - player_width/2
        if self.player_ball_loc[1][2] <= 56*self.scale - player_height/2:
            self.player_ball_loc[1][2] = 56*self.scale - player_height/2
        if self.player_ball_loc[1][2] >= 861*self.scale - player_height/2:
            self.player_ball_loc[1][2] = 861*self.scale - player_height/2

    def keeper_movement(self,keys_pressed):

        if keys_pressed[pygame.K_z]:
            self.player_ball_loc[0][-2] -= keeper_vel
        if keys_pressed[pygame.K_x]:
            self.player_ball_loc[0][-2] += keeper_vel
        if keys_pressed[pygame.K_c]:
            self.player_ball_loc[1][-2] -= keeper_vel
        if keys_pressed[pygame.K_v]:
            self.player_ball_loc[1][-2] += keeper_vel

        if self.player_ball_loc[0][-2] <= 307*self.scale+keeper_width:
            self.player_ball_loc[0][-2] = 307*self.scale+keeper_width
        if self.player_ball_loc[0][-2] >= 923*self.scale:
            self.player_ball_loc[0][-2] = 923*self.scale
        if self.player_ball_loc[1][-2] <= 40:
            self.player_ball_loc[1][-2] = 40
        if self.player_ball_loc[1][-2] >= 283*self.scale:
            self.player_ball_loc[1][-2] = 283*self.scale

    def line_out(self):
        # Check if ball goes out of bounds
        if self.player_ball_loc[0][-1] <= 104*self.scale or self.player_ball_loc[0][-1] >= 1166*self.scale \
            or self.player_ball_loc[1][-1] <= 45*self.scale or self.player_ball_loc[1][-1] >= 862*self.scale:
            return True
    def defender_block(self):
        # Check if defender possesses the ball
        if np.hypot(self.player_ball_loc[0][2]+keeper_width/2-self.player_ball_loc[0][-1]-ball_width/2,
                  self.player_ball_loc[1][2]+keeper_height/2-self.player_ball_loc[1][-1]-ball_height/2) < 20:
            return True
    def keeper_block(self):
        if np.hypot(self.player_ball_loc[0][-2]+keeper_width/2-self.player_ball_loc[0][-1]-ball_width/2,
                  self.player_ball_loc[1][-2]+keeper_height/2-self.player_ball_loc[1][-1]-ball_height/2) < 13:
            return True
    
    def goal(self):
        if self.player_ball_loc[1][-1] <= 50*self.scale and self.player_ball_loc[0][-1] >= 584*self.scale \
            and self.player_ball_loc[0][-1] <= 696*self.scale:
                return True

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mx, self.my = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys_pressed = pygame.key.get_pressed()
        
        # 2. move/pass
        self.possess_ball()
        self.player1_movement(keys_pressed)
        self.player2_movement(keys_pressed)
        self.defender_movement()
        self.keeper_movement(keys_pressed)
        self.pass_shoot(keys_pressed)
        
        # 3. check if game over
        game_over = False
        if self.goal() or self.line_out() or self.keeper_block() or self.defender_block():
            game_over = True
            return game_over, self.goal(), self.line_out(), self.keeper_block(), self.defender_block()
        
        # 4. update ui and clock
        self._update_ui()
        self.clock.tick(FPS)

        # 5. return game over, goal
        return game_over, self.goal(), self.line_out(), self.keeper_block(), self.defender_block()

if __name__ == '__main__':
    game = Football_Game(w=1024,h=720)
    
    # game loop
    while True:
        game_over, goal, line_out, keeper_block, defender_block = game.play_step()
        
        if game_over == True:
            break
    
    if goal:
        print('Congratulations!')
    else:
        print('Game Over')
        
        
    pygame.quit()