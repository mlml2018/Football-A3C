import pygame
import numpy as np

class Football_Game_Visualisation:
    
    def __init__(self, w=1130, h=760):
        import os
        ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')

        self.ballImg = pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_DIR, 'football.png')), (8, 8))
        self.playerImg = pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_DIR, 'player.png')), (23, 23))
        self.defenderImg = pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_DIR, 'defender.png')), (23, 23))
        self.goalImg = pygame.image.load(os.path.join(ASSETS_DIR, 'goal-box-with-net.png'))

        self.w = w
        self.h = h

        self.FPS = 60
        # Player and Ball Sizes
        self.player_width = 23
        self.player_height = 23
        self.ball_width = 8
        self.ball_height = 8
        # Player vel
        self.player_vel = int(3*2)
        self.defender_vel = int(2.4*2)
        # Ball vel
        self.passing_vel = 4.5*2

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Football')
        self.clock = pygame.time.Clock()  # Initialize clock as None
        self.reset()
    
    def reset(self):
        # init game state
        # x coordinate, y coordinate, orientation, player that passed ball
        player_1x = np.random.randint(700,1001)
        player_2x = np.random.randint(700,1001)
        player_1y = np.random.randint(60,self.h/2-self.player_height+1)
        player_2y = np.random.randint(self.h/2+self.player_height,701)
        defender_1x = np.random.randint(50,251)
        defender_2x = np.random.randint(50,251)
        defender_1y = np.random.randint(60,701)
        defender_2y = np.random.randint(60,701)
        ball_possess_boolean = np.random.randint(0,2)
        if ball_possess_boolean:
            ball_x = player_1x + self.player_width//2
            ball_y = player_1y + self.player_height//2
        else:
            ball_x = player_2x + self.player_width//2
            ball_y = player_2y + self.player_height//2
        self.player_ball_loc = np.array([[player_1x, player_2x, defender_1x, defender_2x, ball_x],
                       [player_1y, player_2y, defender_1y, defender_2y, ball_y],
                       [0, 0, 0, 0, 0],
                       [False, False, False, False, False]])    
        
        # mx (mouse x coordinate), my, speed x, speed y, activation
        # speed x, speed y, ball passed boolean
        self.pass_info = np.array([0, 0, False])
        self.possess = np.array([False,False,False,False])

        state = self.player_ball_loc[:2,:].T.flatten()
        return state / 1000.0

        
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
        self.display.blit(pygame.transform.scale(pygame.transform.rotate(self.goalImg, 90), (50, 100)), (-5, 330))
        self.display.blit(pygame.transform.scale(pygame.transform.rotate(self.goalImg, 270), (50, 100)), (1085, 330))

        # Draw players and ball
        self.display.blit(pygame.transform.flip(self.playerImg, True, False), (self.player_ball_loc[0][0], self.player_ball_loc[1][0]))
        self.display.blit(pygame.transform.flip(self.playerImg, True, False), (self.player_ball_loc[0][1], self.player_ball_loc[1][1]))
        self.display.blit(self.defenderImg, (self.player_ball_loc[0][2], self.player_ball_loc[1][2]))
        self.display.blit(self.defenderImg, (self.player_ball_loc[0][3], self.player_ball_loc[1][3]))
        self.display.blit(self.ballImg, (self.player_ball_loc[0][-1], self.player_ball_loc[1][-1]))

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
                if np.hypot(self.player_ball_loc[0][i]+12-self.player_ball_loc[0][-1],
                            self.player_ball_loc[1][i]+12-self.player_ball_loc[1][-1]) < 15 and not self.player_ball_loc[-1][i]:
                    self.pass_info[-1] = False
                    self.possess = np.array([k==i for k in range(len(self.possess))])
                    for j in range(len(self.player_ball_loc[0])-1):
                            self.player_ball_loc[-1][j] = False
                    self.player_ball_loc[0][-1] = self.player_ball_loc[0][i] + 20
                    self.player_ball_loc[1][-1] = self.player_ball_loc[1][i] + 20

    def get_dx_dy(self, move_action, vel):
        if move_action == 0:
            return 0, 0
        angle = (move_action - 1) * np.pi / 4
        dx = np.sin(angle) * vel
        dy = -np.cos(angle) * vel
        return dx, dy

    def pass_ball(self, action):
        pass_action = action[2]
        for i in range(2):
            if np.hypot(self.player_ball_loc[0][i]+10-self.player_ball_loc[0][-1],self.player_ball_loc[1][i]+25-self.player_ball_loc[1][-1]) < 20:    
                if pass_action > 0:  # pass
                    dx, dy = self.get_dx_dy(pass_action, self.passing_vel)
                    self.pass_info[0] = dx
                    self.pass_info[1] = dy
                    self.player_ball_loc[-1][i] = True
                    self.pass_info[-1] = True
        if self.pass_info[-1]:
            self.possess = np.array([False,False,False,False])
            self.player_ball_loc[0][-1] += self.pass_info[0]
            self.player_ball_loc[1][-1] += self.pass_info[1]

    def player1_movement(self, action):
        p1_action = action[0]
        vel = self.player_vel*3/5 if self.possess[0] else self.player_vel
        dx, dy = self.get_dx_dy(p1_action, vel)
        
        # Update player location
        self.player_ball_loc[0][0] += dx
        self.player_ball_loc[1][0] += dy
    
        if self.player_ball_loc[0][0] <= 40 - self.player_width:
            self.player_ball_loc[0][0] = 40 - self.player_width
        if self.player_ball_loc[0][0] >= 1090:
            self.player_ball_loc[0][0] = 1090
        if self.player_ball_loc[1][0] <= 40 - self.player_height:
            self.player_ball_loc[1][0] = 40 - self.player_height
        if self.player_ball_loc[1][0] >= 720:
            self.player_ball_loc[1][0] = 720
    
    def player2_movement(self, action):
        p2_action = action[1]
        vel = self.player_vel*3/5 if self.possess[1] else self.player_vel
        dx, dy = self.get_dx_dy(p2_action, vel)

        # Update player location
        self.player_ball_loc[0][1] += dx
        self.player_ball_loc[1][1] += dy


        if self.player_ball_loc[0][1] <= 40 - self.player_width:
            self.player_ball_loc[0][1] = 40 - self.player_width
        if self.player_ball_loc[0][1] >= 1090:
            self.player_ball_loc[0][1] = 1090
        if self.player_ball_loc[1][1] <= 40 - self.player_height:
            self.player_ball_loc[1][1] = 40 - self.player_height
        if self.player_ball_loc[1][1] >= 720:
            self.player_ball_loc[1][1] = 720

    # def defender1_movement(self):
    #     # Distance between ball and defender
    #     if self.player_ball_loc[-1][2]:
    #         dx = 150-self.player_ball_loc[0][2]
    #         dy = self.h/2+10-self.player_ball_loc[0][2]
    #     else:
    #         dx = (self.player_ball_loc[0][-1]+4-self.player_ball_loc[0][2]-20)
    #         dy = (self.player_ball_loc[1][-1]+4-self.player_ball_loc[1][2]-20)
        
    #     d = np.hypot(dx,dy)

    #     self.player_ball_loc[0][2] += dx/d * self.defender_vel
    #     self.player_ball_loc[1][2] += dy/d * self.defender_vel


    #     if self.player_ball_loc[0][2] <= 40 - self.player_width:
    #         self.player_ball_loc[0][2] = 40 - self.player_width
    #     if self.player_ball_loc[0][2] >= 1090:
    #         self.player_ball_loc[0][2] = 1090
    #     if self.player_ball_loc[1][2] <= 40 - self.player_height:
    #         self.player_ball_loc[1][2] = 40 - self.player_height
    #     if self.player_ball_loc[1][2] >= 720:
    #         self.player_ball_loc[1][2] = 720
    def defender1_movement(self):
        # Distance between ball and defender
        if self.player_ball_loc[-1][2]:
            # dx = 150-self.player_ball_loc[0][2]
            # dy = self.h/2+100-self.player_ball_loc[0][2]
            dx, dy = 0, 0
        else:
            dx = (self.player_ball_loc[0][-1]+4-self.player_ball_loc[0][2]-20)
            dy = (self.player_ball_loc[1][-1]+4-self.player_ball_loc[1][2]-20)
        
        d = np.hypot(dx,dy)
        if d > 0:
            self.player_ball_loc[0][2] += dx/d * self.defender_vel
            self.player_ball_loc[1][2] += dy/d * self.defender_vel


        if self.player_ball_loc[0][2] <= 40 - self.player_width:
            self.player_ball_loc[0][2] = 40 - self.player_width
        if self.player_ball_loc[0][2] >= 1090:
            self.player_ball_loc[0][2] = 1090
        if self.player_ball_loc[1][2] <= 40 - self.player_height:
            self.player_ball_loc[1][2] = 40 - self.player_height
        if self.player_ball_loc[1][2] >= 720:
            self.player_ball_loc[1][2] = 720
    
    def defender2_movement(self):
        # Distance between ball and defender
        if self.player_ball_loc[-1][3]:
            # dx = 150-self.player_ball_loc[0][2]
            # dy = self.h/2+100-self.player_ball_loc[0][2]
            dx, dy = 0, 0
        else:
            dx = (self.player_ball_loc[0][-1]+4-self.player_ball_loc[0][3]-20)
            dy = (self.player_ball_loc[1][-1]+4-self.player_ball_loc[1][3]-20)
        
        d = np.hypot(dx,dy)
        if d > 0:
            self.player_ball_loc[0][3] += dx/d * self.defender_vel *0.7
            self.player_ball_loc[1][3] += dy/d * self.defender_vel *0.7

        if self.player_ball_loc[0][3] <= 40 - self.player_width:
            self.player_ball_loc[0][3] = 40 - self.player_width
        if self.player_ball_loc[0][3] >= 1090:
            self.player_ball_loc[0][3] = 1090
        if self.player_ball_loc[1][3] <= 40 - self.player_height:
            self.player_ball_loc[1][3] = 40 - self.player_height
        if self.player_ball_loc[1][3] >= 720:
            self.player_ball_loc[1][3] = 720

    def line_out(self):
        # Check if ball goes out of bounds
        if self.player_ball_loc[0][-1] <= 40-11 or self.player_ball_loc[0][-1] >= 1090+11 \
            or self.player_ball_loc[1][-1] <= 40-11 or self.player_ball_loc[1][-1] >= 720+11:
            return True
    
    def defender_ball(self):
        if self.possess[-1] or self.possess[-2]:
            return True
    
    def goal(self):
        if self.player_ball_loc[0][-1] <= 40-self.ball_width and self.player_ball_loc[1][-1] >= self.h/2 - 50 \
            and self.player_ball_loc[1][-1] <= self.h/2 + 50:
            return True
    
    
    def own_goal(self):
        if self.player_ball_loc[0][-1] >= 1090 and self.player_ball_loc[1][-1] >= self.h/2 - 50 \
            and self.player_ball_loc[1][-1] <= self.h/2 + 50:
                return True

    def play_step(self,action):
        pygame.init()
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Observe current state, check possession of ball and reward accordingly
        reward = 0

        # 3. move/pass
        self.possess_ball()
        self.player1_movement(action)
        self.player2_movement(action)
        
        # PHASE 1 CURRICULUM: Disable defenders visually
        # self.defender1_movement()
        # self.defender2_movement()
        
        self.pass_ball(action)
        
        # 4. check if game over
        game_over = False
        state = self.player_ball_loc[:2,:].T.flatten()
        next_state = state / 1000.0
        
        # PHASE 1 CURRICULUM: Ignore defender ball visually
        if self.goal() or self.own_goal() or self.line_out(): # or self.defender_ball():
            game_over = True
            if self.goal():
                reward += 10
                return next_state, reward, game_over
            elif self.own_goal():
                reward -= 10
                return next_state, reward, game_over
            else:
                reward -= 5
                return next_state, reward, game_over

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.FPS)
        
        # 6. return next state, reward, game over boolean
        return next_state, reward, game_over

if __name__ == '__main__':
    game = Football_Game_Visualisation()
    
    # game loop
    while True:
        next_state, reward, game_over = game.play_step(action = [0,0,0,0])
        
        if game_over:
            break
        
    print('Final Score', reward)
        
        
    pygame.quit()