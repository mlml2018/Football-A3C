# import pygame
# import numpy as np

# pygame.init()
# screen_width = 1130
# screen_height = 760
# screen = pygame.display.set_mode((screen_width, screen_height))

# player_width = 23
# player_height = 23
# playerImg = pygame.transform.scale(pygame.image.load('player.png'), (player_width, player_height))
# goalImg = pygame.image.load('goal-box-with-net.png')

# FPS = 60

# def main():
#     clock = pygame.time.Clock()
#     running = True

#     while running:
#         clock.tick(FPS)
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#         screen.fill((0, 128, 0))
#         screen.blit(pygame.transform.scale(pygame.transform.rotate(goalImg, 90), (50, 100)), (-5, 330))
#         screen.blit(pygame.transform.scale(pygame.transform.rotate(goalImg, 270), (50, 100)), (1085, 330))
#         pygame.draw.line(screen, (255, 255, 255), (40, 40), (1090, 40), 1)
#         pygame.draw.line(screen, (255, 255, 255), (1090, 40), (1090, 720), 1)
#         pygame.draw.line(screen, (255, 255, 255), (1090, 720), (40, 720), 1)
#         pygame.draw.line(screen, (255, 255, 255), (40, 720), (40, 40), 1)

#         # penalty area
#         # Left side
#         pygame.draw.line(screen, (255, 255, 255), (40, 285), (95, 285), 1)
#         pygame.draw.line(screen, (255, 255, 255), (95, 285), (95, 475), 1)
#         pygame.draw.line(screen, (255, 255, 255), (95, 475), (40, 475), 1)

#         pygame.draw.line(screen, (255, 255, 255), (40, 175), (205, 175), 1)
#         pygame.draw.line(screen, (255, 255, 255), (205, 175), (205, 585), 1)
#         pygame.draw.line(screen, (255, 255, 255), (205, 585), (40, 585), 1)

#         pygame.draw.circle(screen, (255, 255, 255), (150, 380), 4)
#         pygame.draw.arc(screen, (255, 255, 255), [58, 288, 184, 184], np.pi*2- np.arccos(55/92), np.pi*2+ np.arccos(55/92))

#         # Right side
#         pygame.draw.line(screen, (255, 255, 255), (screen_width-40, 285), (screen_width-95, 285), 1)
#         pygame.draw.line(screen, (255, 255, 255), (screen_width-95, 285), (screen_width-95, 475), 1)
#         pygame.draw.line(screen, (255, 255, 255), (screen_width-95, 475), (screen_width-40, 475), 1)

#         pygame.draw.line(screen, (255, 255, 255), (screen_width-40, 175), (screen_width-205, 175), 1)
#         pygame.draw.line(screen, (255, 255, 255), (screen_width-205, 175), (screen_width-205, 585), 1)
#         pygame.draw.line(screen, (255, 255, 255), (screen_width-205, 585), (screen_width-40, 585), 1)

#         pygame.draw.circle(screen, (255, 255, 255), (screen_width-150, 380), 4)
#         pygame.draw.arc(screen, (255, 255, 255), [screen_width-58-184, 288, 184, 184], np.pi - np.arccos(55/92), np.pi + np.arccos(55/92))

#         # # Centre circle
#         pygame.draw.arc(screen, (255, 255, 255), [screen_width/2-92, screen_height/2-92, 184, 184], 0, 2*np.pi,1)
#         pygame.draw.circle(screen, (255, 255, 255), (screen_width/2, screen_height/2), 4)
#         pygame.draw.line(screen, (255,255,255), (screen_width/2,40), (screen_width/2,screen_height-40))

#         pygame.display.update()

# if __name__ == "__main__":

#     main()

import numpy as np

# Example 4x4 matrix
matrix = np.array([
    [1, 2, 3, 4],  # Row 1
    [5, 6, 7, 8],  # Row 2
    [9, 10, 11, 12],  # Row 3
    [13, 14, 15, 16]  # Row 4
])

# # Extract the first two rows
# first_two_rows = matrix[:2, :]  # Shape (2, 4)

# # Interleave elements row-wise
# result = first_two_rows.T.flatten()  # Shape (8,)

# print(result)

print(matrix[:2,:].T.flatten())
print(matrix)

print(np.random.random_integers(0,1).dtype)
print(7//2)
