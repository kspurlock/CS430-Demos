# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 22:10:39 2021

@author: kylei
"""

'''
Contains classes and functions for
implementing custom sprites in snake game.
'''


import pygame
import numpy as np

RED = (255,0,0)
GREEN = (0,255,0)
WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREY = (128,128,128)
LGREY = (170,170,170)

class Col_block(pygame.sprite.Sprite):
    def __init__(self, x, y, rectx, recty):
        super(Col_block, self).__init__()
        self.surf = pygame.Surface((rectx, recty))
        self.surf.fill(BLUE)
        self.rect = self.surf.get_rect(center = (x, y))
    
    def update(self, x, y, rectx, recty):
        self.surf = pygame.Surface((rectx, recty))
        self.surf.fill(BLUE)
        self.rect = self.surf.get_rect(center = (x,y))
        
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surf = pygame.Surface((25, 25)) ##Size of the sprites surface
        self.unrotated_surf = self.surf
        self.surf.fill(WHITE) #Colour of the sprite

        self.rect = self.surf.get_rect(center = ((np.random.randint(2,18)*25)-12.5, (np.random.randint(2, 18)*25)-12.5)) #The starting position of the head sprite
        self.body = [Body(self.rect.x-25, self.rect.y), Body(self.rect.x-50, self.rect.y), Body(self.rect.x-75, self.rect.y)] #Body to hold the length of the snake
        
        #Elements for collision blocks
        self.anchorfront = self.rect.midright
        self.anchorleft = self.rect.midtop
        self.anchorright = self.rect.midbottom
        self.col_blockF = Col_block(self.anchorfront[0]+25, self.anchorfront[1], 25, 25)
        self.col_blockL = Col_block(self.anchorleft[0], self.anchorleft[1]-25, 25,25)
        self.col_blockR = Col_block(self.anchorleft[0], self.anchorleft[1]+25, 25, 25)
    
    def update(self, direction, change_dir):
        #Performing movement
        if direction == 'UP':
            self.rect.move_ip(0, -25)
        elif direction == 'DOWN':
            self.rect.move_ip(0, 25)
        elif direction == 'LEFT':
            self.rect.move_ip(-25, 0)
        elif direction == 'RIGHT':
            self.rect.move_ip(25, 0)
        
        #Changing anchor points after move
        if direction == 'UP':
            self.anchorfront = self.rect.midtop
            self.anchorleft = self.rect.midleft
            self.anchorright = self.rect.midright
        elif direction == 'DOWN':
            self.anchorfront = self.rect.midbottom
            self.anchorleft = self.rect.midright
            self.anchorright = self.rect.midleft
        elif direction == 'LEFT':
            self.anchorfront = self.rect.midleft
            self.anchorright = self.rect.midtop
            self.anchorleft = self.rect.midbottom
        elif direction == 'RIGHT':
            self.anchorfront = self.rect.midright
            self.anchorleft = self.rect.midtop
            self.anchorright = self.rect.midbottom
        
        self.update_cds(direction)
        return change_dir
        
    def update_cds(self, direction):
        if direction == 'UP':
            self.col_blockF.update(self.anchorfront[0], self.anchorfront[1]-12.5, 25, 25)
            self.col_blockL.update(self.anchorleft[0]-12.5, self.anchorfront[1]+12.5, 25, 25)
            self.col_blockR.update(self.anchorright[0]+12.5, self.anchorfront[1]+12.5, 25, 25)
        elif direction == 'DOWN':
            self.col_blockF.update(self.anchorfront[0], self.anchorfront[1]+12.5, 25, 25)
            self.col_blockR.update(self.anchorright[0]-12.5, self.anchorright[1], 25, 25)
            self.col_blockL.update(self.anchorleft[0]+12.5, self.anchorright[1],25,25)
        elif direction == 'LEFT':
            self.col_blockF.update(self.anchorfront[0]-12.5, self.anchorfront[1], 25, 25)
            self.col_blockL.update(self.anchorleft[0], self.anchorleft[1]+12.5, 25, 25)
            self.col_blockR.update(self.anchorright[0], self.anchorright[1]-12.5, 25, 25)
        elif direction == 'RIGHT':
            self.col_blockF.update(self.anchorfront[0]+12.5, self.anchorfront[1], 25, 25)
            self.col_blockL.update(self.anchorleft[0], self.anchorleft[1]-12.5, 25, 25)
            self.col_blockR.update(self.anchorright[0], self.anchorright[1]+12.5, 25, 25)
    
class Food(pygame.sprite.Sprite):
    '''Handles the sprite of the food object'''
    def __init__(self):
        super(Food, self).__init__()
        self.surf = pygame.Surface((25, 25))
        self.surf.fill(RED)
        self.rect = self.surf.get_rect(center = ((np.random.randint(2,18)*25)-12.5, (np.random.randint(2, 18)*25)-12.5))
    
    def update(self):
        self.rect.center = ((np.random.randint(2,18)*25)-12.5, (np.random.randint(2, 18)*25)-12.5)

class Body(pygame.sprite.Sprite):
    '''Handles the sprites of the snakes body'''
    def __init__(self, x, y):
        super(Body, self).__init__()
        self.surf = pygame.Surface((25,25))
        self.surf.fill((100,100,100))
        self.rect = self.surf.get_rect(center = (x,y))
        
class Wall(pygame.sprite.Sprite):
    '''Handles the sprites of the boundary walls'''
    def __init__(self, x, y, rectx, recty):
        super(Wall, self).__init__()
        self.surf = pygame.Surface((rectx,recty))
        self.surf.fill(GREY)
        self.rect = self.surf.get_rect(center = (x, y))
