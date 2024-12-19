import pygame
import numpy as np

class Football_Game:

    def __init__(self, w=1130, h=760):

        self.w = w
        self.h = h
        # Player and Ball Sizes
        self.player_width = 23
        self.player_height = 23
        self.ball_width = 8
        self.ball_height = 8
        # Player vel
        self.player_vel = int(3*2)
        #self.defender_vel = int(2.775)
        self.defender_vel = int(2.4*2)
        # Ball vel
        self.passing_vel = 4.5*2
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
        self.ball_possess_boolean = np.random.randint(0,2)
        if self.ball_possess_boolean:
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

        return self.player_ball_loc[:2,:].T.flatten()
    

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

    def pass_ball(self, action):
        for i in range(2):
            if np.hypot(self.player_ball_loc[0][i]+10-self.player_ball_loc[0][-1],self.player_ball_loc[1][i]+25-self.player_ball_loc[1][-1]) < 20:    
                if action[-4]:  # pass
                    self.pass_info[0] = np.sin(action[-1])*self.passing_vel
                    self.pass_info[1] = np.cos(action[-1])*self.passing_vel
                    self.player_ball_loc[-1][i] = True
                    self.pass_info[-1] = True
            # elif np.hypot(self.player_ball_loc[0][i+2]+10-self.player_ball_loc[0][-1],self.player_ball_loc[1][i+2]+25-self.player_ball_loc[1][-1]) < 20:
            #     dist = np.hypot(self.w-40-self.player_ball_loc[0][-1], self.h/2-self.player_ball_loc[1][-1])
            #     self.pass_info[0] = (self.w-40-self.player_ball_loc[0][-1])/dist*self.passing_vel
            #     self.pass_info[1] = (self.h/2-self.player_ball_loc[1][-1])/dist*self.passing_vel
            #     self.player_ball_loc[-1][i+2] = True
            #     self.pass_info[-1] = True
        if self.pass_info[-1]:
            self.possess = np.array([False,False,False,False])
            self.player_ball_loc[0][-1] += self.pass_info[0]
            self.player_ball_loc[1][-1] -= self.pass_info[1]

    def player1_movement(self, action):
        possess_ball = self.possess[0]
        # Update player location
        if possess_ball:
            self.player_ball_loc[0][0] += np.sin(action[-3]) * self.player_vel*3/5
            self.player_ball_loc[1][0] -= np.cos(action[-3]) * self.player_vel*3/5
        else:
            self.player_ball_loc[0][0] += np.sin(action[-3]) * self.player_vel
            self.player_ball_loc[1][0] -= np.cos(action[-3]) * self.player_vel
        
        if self.player_ball_loc[0][0] <= 40 - self.player_width:
            self.player_ball_loc[0][0] = 40 - self.player_width
        if self.player_ball_loc[0][0] >= 1090:
            self.player_ball_loc[0][0] = 1090
        if self.player_ball_loc[1][0] <= 40 - self.player_height:
            self.player_ball_loc[1][0] = 40 - self.player_height
        if self.player_ball_loc[1][0] >= 720:
            self.player_ball_loc[1][0] = 720
    
    def player2_movement(self, action):
        possess_ball = self.possess[1]

        # Update player location
        if possess_ball:
            self.player_ball_loc[0][1] += np.sin(action[-2]) * self.player_vel*3/5
            self.player_ball_loc[1][1] -= np.cos(action[-2]) * self.player_vel*3/5
        else:
            self.player_ball_loc[0][1] += np.sin(action[-2]) * self.player_vel
            self.player_ball_loc[1][1] -= np.cos(action[-2]) * self.player_vel


        if self.player_ball_loc[0][1] <= 40 - self.player_width:
            self.player_ball_loc[0][1] = 40 - self.player_width
        if self.player_ball_loc[0][1] >= 1090:
            self.player_ball_loc[0][1] = 1090
        if self.player_ball_loc[1][1] <= 40 - self.player_height:
            self.player_ball_loc[1][1] = 40 - self.player_height
        if self.player_ball_loc[1][1] >= 720:
            self.player_ball_loc[1][1] = 720

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
            self.player_ball_loc[0][3] += dx/d * self.defender_vel*0.7
            self.player_ball_loc[1][3] += dy/d * self.defender_vel*0.7

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
        
    def goal(self):
        if self.player_ball_loc[0][-1] <= 40-self.ball_width and self.player_ball_loc[1][-1] >= self.h/2 - 50 \
            and self.player_ball_loc[1][-1] <= self.h/2 + 50:
            return True

    def defender_ball(self):
        if self.possess[-1] or self.possess[-2]:
            return True
    
    def own_goal(self):
        if self.player_ball_loc[0][-1] >= 1090 and self.player_ball_loc[1][-1] >= self.h/2 - 50 \
            and self.player_ball_loc[1][-1] <= self.h/2 + 50:
                return True

    def play_step(self,action):
        # 1. collect user input
        if not pygame.get_init():
            pygame.init()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Observe current state, check possession of ball and reward accordingly - makes pass not preferrable
        reward = 0

        # reward if pass is successful
        if self.possess[self.ball_possess_boolean]:
            reward += 20

        # if ball close to the goal post increase reward
        ball_to_post = np.hypot(self.player_ball_loc[0][-1]-40, self.player_ball_loc[1][-1]-self.h/2)
        if ball_to_post > 0:
            reward += 100/ball_to_post
        player1_to_post = np.hypot(self.player_ball_loc[0][0]-40, self.player_ball_loc[1][0]-self.h/2)
        if player1_to_post > 0:
            reward += 100/player1_to_post
        player2_to_post = np.hypot(self.player_ball_loc[0][1]-40, self.player_ball_loc[1][1]-self.h/2)
        if player2_to_post > 0:
            reward += 100/player2_to_post
        
        # 3. move/pass
        self.possess_ball()
        self.player1_movement(action)
        self.player2_movement(action)
        self.defender1_movement()
        self.defender2_movement()
        self.pass_ball(action)
        
        # 4. check if game over
        game_over = False
        next_state = self.player_ball_loc[:2,:].T.flatten()
        if self.goal() or self.own_goal() or self.line_out() or self.defender_ball():
            game_over = True
            if self.goal():
                reward += 30
                return next_state, reward, game_over
            elif self.own_goal():
                reward -= 10
                return next_state, reward, game_over
            elif self.line_out():
                reward -= 2
            else: # line out or defender ball
                reward -= 20
                return next_state, reward, game_over
        
        # 5. return next state, reward, game over boolean
        return next_state, reward, game_over
