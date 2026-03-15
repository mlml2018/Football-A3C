import pygame
import numpy as np

pygame.init()
#font = pygame.font.Font('arial.ttf', 25)


# Player and Ball Sizes
player_width = 23
player_height = 23
ball_width = 8
ball_height = 8

# Player, Ball, Goal Post Images
ballImg = pygame.transform.scale(pygame.image.load('football.png'), (ball_width, ball_height))
playerImg = pygame.transform.scale(pygame.image.load('player.png'), (player_width, player_height))
defenderImg = pygame.transform.scale(pygame.image.load('defender.png'), (player_width, player_height))
goalImg = pygame.image.load('goal-box-with-net.png')

# Player vel
player_vel = int(3)
defender_vel = int(2.775)
defender_vel = int(2)


# Ball vel
passing_vel = 4

FPS = 60


class Football_Game():

    def __init__(self, w=1130, h=760):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Football')
        self.clock = pygame.time.Clock()
        self.reset()

    
    def reset(self):
        # init game state
        # x coordinate, y coordinate, orientation, player that passed ball
        # self.player_ball_loc = np.array([[1000, 1000, 150, 150, 1012],
        #                [680, 80, self.h/2+150, self.h/2-150, 690],
        #                [0, 0, 0, 0, 0],
        #                [False, False, False, False, False]])
        player_1x = np.random.randint(700,1001)
        player_2x = np.random.randint(700,1001)
        player_1y = np.random.randint(60,self.h/2-player_height+1)
        player_2y = np.random.randint(self.h/2+player_height,701)
        defender_1x = np.random.randint(50,251)
        defender_2x = np.random.randint(50,251)
        defender_1y = np.random.randint(60,701)
        defender_2y = np.random.randint(60,701)
        ball_possess_boolean = np.random.randint(0,2)
        if ball_possess_boolean:
            ball_x = player_1x + player_width//2
            ball_y = player_1y + player_height//2
        else:
            ball_x = player_2x + player_width//2
            ball_y = player_2y + player_height//2
        self.player_ball_loc = np.array([[player_1x, player_2x, defender_1x, defender_2x, ball_x],
                       [player_1y, player_2y, defender_1y, defender_2y, ball_y],
                       [0, 0, 0, 0, 0],
                       [False, False, False, False, False]])
        
        # mx (mouse x coordinate), my, speed x, speed y, activation
        # speed x, speed y, ball passed boolean
        self.pass_info = np.array([0, 0, False])
        self.mx, self.my = 40, self.h/2
        self.possess = np.array([False,False,False,False])
        
    def _update_ui(self):

        self.display.fill((0, 128, 0))
        
        pygame.draw.line(self.display, (255, 255, 255), (40, 40), (1090, 40), 1)
        pygame.draw.line(self.display, (255, 255, 255), (1090, 40), (1090, 720), 1)
        pygame.draw.line(self.display, (255, 255, 255), (1090, 720), (40, 720), 1)
        pygame.draw.line(self.display, (255, 255, 255), (40, 720), (40, 40), 1)

        # penalty area
        # Left side
        pygame.draw.line(self.display, (255, 255, 255), (40, 285), (95, 285), 1)
        pygame.draw.line(self.display, (255, 255, 255), (95, 285), (95, 475), 1)
        pygame.draw.line(self.display, (255, 255, 255), (95, 475), (40, 475), 1)

        pygame.draw.line(self.display, (255, 255, 255), (40, 175), (205, 175), 1)
        pygame.draw.line(self.display, (255, 255, 255), (205, 175), (205, 585), 1)
        pygame.draw.line(self.display, (255, 255, 255), (205, 585), (40, 585), 1)

        pygame.draw.circle(self.display, (255, 255, 255), (150, 380), 4)
        pygame.draw.arc(self.display, (255, 255, 255), [58, 288, 184, 184], np.pi*2- np.arccos(55/92), np.pi*2+ np.arccos(55/92))

        # Right side
        pygame.draw.line(self.display, (255, 255, 255), (self.w-40, 285), (self.w-95, 285), 1)
        pygame.draw.line(self.display, (255, 255, 255), (self.w-95, 285), (self.w-95, 475), 1)
        pygame.draw.line(self.display, (255, 255, 255), (self.w-95, 475), (self.w-40, 475), 1)

        pygame.draw.line(self.display, (255, 255, 255), (self.w-40, 175), (self.w-205, 175), 1)
        pygame.draw.line(self.display, (255, 255, 255), (self.w-205, 175), (self.w-205, 585), 1)
        pygame.draw.line(self.display, (255, 255, 255), (self.w-205, 585), (self.w-40, 585), 1)

        pygame.draw.circle(self.display, (255, 255, 255), (self.w-150, 380), 4)
        pygame.draw.arc(self.display, (255, 255, 255), [self.w-58-184, 288, 184, 184], np.pi - np.arccos(55/92), np.pi + np.arccos(55/92))

        # # Centre circle
        pygame.draw.arc(self.display, (255, 255, 255), [self.w/2-92, self.h/2-92, 184, 184], 0, 2*np.pi,1)
        pygame.draw.circle(self.display, (255, 255, 255), (self.w/2, self.h/2), 4)
        pygame.draw.line(self.display, (255,255,255), (self.w/2,40), (self.w/2,self.h-40))

        # Draw goal post
        self.display.blit(pygame.transform.scale(pygame.transform.rotate(goalImg, 90), (50, 100)), (-5, 330))
        self.display.blit(pygame.transform.scale(pygame.transform.rotate(goalImg, 270), (50, 100)), (1085, 330))

        # Draw players and ball
        self.display.blit(pygame.transform.flip(playerImg, True, False), (self.player_ball_loc[0][0], self.player_ball_loc[1][0]))
        self.display.blit(pygame.transform.flip(playerImg, True, False), (self.player_ball_loc[0][1], self.player_ball_loc[1][1]))
        self.display.blit(defenderImg, (self.player_ball_loc[0][2], self.player_ball_loc[1][2]))
        self.display.blit(defenderImg, (self.player_ball_loc[0][3], self.player_ball_loc[1][3]))
        self.display.blit(ballImg, (self.player_ball_loc[0][-1], self.player_ball_loc[1][-1]))


        pygame.display.flip()
    
    def possess_ball(self):
        for i in range(len(self.player_ball_loc[0])-1):
            if i<2:
                if np.hypot(self.player_ball_loc[0][i]+10-self.player_ball_loc[0][-1],
                            self.player_ball_loc[1][i]+25-self.player_ball_loc[1][-1]) < 20 and not self.player_ball_loc[-1][i]:
                    self.pass_info[-1] = False
                    self.possess = np.array([k==i for k in range(len(self.possess))])
                    for j in range(len(self.player_ball_loc[0])-1):
                        self.player_ball_loc[-1][j] = False
                    self.player_ball_loc[0][-1] = self.player_ball_loc[0][i]
                    self.player_ball_loc[1][-1] = self.player_ball_loc[1][i] + 20
            else:
                if np.hypot(self.player_ball_loc[0][i]+10-self.player_ball_loc[0][-1],
                            self.player_ball_loc[1][i]+25-self.player_ball_loc[1][-1]) < 25 and not self.player_ball_loc[-1][i]:
                    self.pass_info[-1] = False
                    self.possess = np.array([k==i for k in range(len(self.possess))])
                    for j in range(len(self.player_ball_loc[0])-1):
                            self.player_ball_loc[-1][j] = False
                    self.player_ball_loc[0][-1] = self.player_ball_loc[0][i] + 20
                    self.player_ball_loc[1][-1] = self.player_ball_loc[1][i] + 20

    def pass_shoot(self, keys_pressed):
        for i in range(2):
            if np.hypot(self.player_ball_loc[0][i]+10-self.player_ball_loc[0][-1],self.player_ball_loc[1][i]+25-self.player_ball_loc[1][-1]) < 20:    
                dist = np.hypot(self.mx-self.player_ball_loc[0][-1], self.my-self.player_ball_loc[1][-1])
                if keys_pressed[pygame.K_SPACE] and dist > 0:  # pass
                    self.pass_info[0] = (self.mx-self.player_ball_loc[0][-1])/dist*passing_vel
                    self.pass_info[1] = (self.my-self.player_ball_loc[1][-1])/dist*passing_vel
                    self.player_ball_loc[-1][i] = True
                    self.pass_info[-1] = True
            elif np.hypot(self.player_ball_loc[0][i+2]+10-self.player_ball_loc[0][-1],self.player_ball_loc[1][i+2]+25-self.player_ball_loc[1][-1]) < 20:
                dist = np.hypot(self.w-40-self.player_ball_loc[0][-1], self.h/2-self.my-self.player_ball_loc[1][-1])
                self.pass_info[0] = (self.w-40-self.player_ball_loc[0][-1])/dist*passing_vel
                self.pass_info[1] = (self.h/2-self.my-self.player_ball_loc[1][-1])/dist*passing_vel
                self.player_ball_loc[-1][i+2] = True
                self.pass_info[-1] = True
        if self.pass_info[-1]:
            self.possess = np.array([False,False,False,False])
            self.player_ball_loc[0][-1] += self.pass_info[0]
            self.player_ball_loc[1][-1] += self.pass_info[1]

    def player1_movement(self, keys_pressed):
        #possess_ball = np.hypot(self.player_ball_loc[0][0]+10-self.player_ball_loc[0][-1],self.player_ball_loc[1][0]+25-self.player_ball_loc[1][-1]) < 20
        possess_ball = self.possess[0]
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
        
        if self.player_ball_loc[0][0] <= 40 - player_width:
            self.player_ball_loc[0][0] = 40 - player_width
        if self.player_ball_loc[0][0] >= 1090:
            self.player_ball_loc[0][0] = 1090
        if self.player_ball_loc[1][0] <= 40 - player_height:
            self.player_ball_loc[1][0] = 40 - player_height
        if self.player_ball_loc[1][0] >= 720:
            self.player_ball_loc[1][0] = 720
    
    def player2_movement(self, keys_pressed):
        #possess_ball = np.hypot(self.player_ball_loc[0][1]+10-self.player_ball_loc[0][-1],self.player_ball_loc[1][1]+25-self.player_ball_loc[1][-1]) < 20
        possess_ball = self.possess[1]
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


        if self.player_ball_loc[0][1] <= 40 - player_width:
            self.player_ball_loc[0][1] = 40 - player_width
        if self.player_ball_loc[0][1] >= 1090:
            self.player_ball_loc[0][1] = 1090
        if self.player_ball_loc[1][1] <= 40 - player_height:
            self.player_ball_loc[1][1] = 40 - player_height
        if self.player_ball_loc[1][1] >= 720:
            self.player_ball_loc[1][1] = 720

    def defender1_movement(self):
        # Distance between ball and defender
        if self.player_ball_loc[-1][2]:
            # dx = 150-self.player_ball_loc[0][2]
            # dy = self.h/2+40-self.player_ball_loc[0][2]
            dx, dy = 0,0
        else:
            dx = (self.player_ball_loc[0][-1]+4-self.player_ball_loc[0][2]-20)
            dy = (self.player_ball_loc[1][-1]+4-self.player_ball_loc[1][2]-20)
        
        d = np.hypot(dx,dy)
        if d > 0:
            self.player_ball_loc[0][2] += dx/d * defender_vel
            self.player_ball_loc[1][2] += dy/d * defender_vel


        if self.player_ball_loc[0][2] <= 40 - player_width:
            self.player_ball_loc[0][2] = 40 - player_width
        if self.player_ball_loc[0][2] >= 1090:
            self.player_ball_loc[0][2] = 1090
        if self.player_ball_loc[1][2] <= 40 - player_height:
            self.player_ball_loc[1][2] = 40 - player_height
        if self.player_ball_loc[1][2] >= 720:
            self.player_ball_loc[1][2] = 720

    def defender2_movement(self):

        if self.player_ball_loc[0][3] <= 40 - player_width:
            self.player_ball_loc[0][3] = 40 - player_width
        if self.player_ball_loc[0][3] >= 1090:
            self.player_ball_loc[0][3] = 1090
        if self.player_ball_loc[1][3] <= 40 - player_height:
            self.player_ball_loc[1][3] = 40 - player_height
        if self.player_ball_loc[1][3] >= 720:
            self.player_ball_loc[1][3] = 720

    def line_out(self):
        # Check if ball goes out of bounds
        if self.player_ball_loc[0][-1] <= 40-11 or self.player_ball_loc[0][-1] >= 1090+11 \
            or self.player_ball_loc[1][-1] <= 40-11 or self.player_ball_loc[1][-1] >= 720+11:
            return True
        
    def goal(self):
        if self.player_ball_loc[0][-1] <= 40-ball_width and self.player_ball_loc[1][-1] >= self.h/2 - 50 \
            and self.player_ball_loc[1][-1] <= self.h/2 + 50:
            return True
    
    
    def own_goal(self):
        if self.player_ball_loc[0][-1] >= 1090 and self.player_ball_loc[1][-1] >= self.h/2 - 50 \
            and self.player_ball_loc[1][-1] <= self.h/2 + 50:
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
        self.defender1_movement()
        self.defender2_movement()
        self.pass_shoot(keys_pressed)
        
        # 3. check if game over
        game_over = False
        if self.goal() or self.own_goal() or self.line_out():
            game_over = True
            return game_over, self.goal(), self.own_goal(), self.line_out()
        
        # 4. update ui and clock
        self._update_ui()
        self.clock.tick(FPS)

        # 5. return game over, goal
        return game_over, self.goal(), self.own_goal(), self.line_out()

if __name__ == '__main__':
    game = Football_Game(w=1130,h=760)
    
    # game loop
    while True:
        game_over, goal, own_goal, line_out = game.play_step()
        
        if game_over == True:
            break
    
    if goal:
        print('Congratulations!')
    elif own_goal:
        print('Lost')
    else:
        print('line out')
        
        
    pygame.quit()