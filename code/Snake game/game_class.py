# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 22:18:43 2021

@author: kylei
"""
import pygame
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT, K_ESCAPE, KEYDOWN
from tensorflow import keras
import math
import numpy as np
import time
import tkinter

from sprites import Player, Food, Wall, Body

'''Declaring constants'''
SCREEN_HEIGHT = 500
SCREEN_WIDTH = 500

RED = (255,0,0)
GREEN = (0,255,0)
WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREY = (128,128,128)
LGREY = (170,170,170)

def end_simulation():
    global tk_end
    tk_end = False


class GameClass():
    def __init__(self, model_inp = None):
        #Defining all necessary sprites
        self.player = Player()
        self.food = Food()
        self.wallL = Wall(12.5, (SCREEN_HEIGHT/2), 25, SCREEN_HEIGHT)
        self.wallU = Wall((SCREEN_WIDTH/2), 12.5, SCREEN_WIDTH, 25)
        self.wallR = Wall((SCREEN_WIDTH)-12.5, SCREEN_HEIGHT/2, 25, SCREEN_HEIGHT)
        self.wallD = Wall((SCREEN_WIDTH/2), SCREEN_HEIGHT-12.5, SCREEN_WIDTH, 25)
        
        #Sprite groups for collisions
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.food)
        
        self.walls = pygame.sprite.Group()
        self.walls.add([self.wallL, self.wallU, self.wallR, self.wallD])
        
        #Vars to hold game state
        self.running = True
        self.snake_die = False
        
        #Metrics of the game
        self.iterations = 0
        self.score_value = 0
        
        #Var to hold the ANN model
        self.model = model_inp
        
    def play_game(self, mode = "ANN", file = None,
                  additional_params = {"FPS":50, "GRID":False}):
        
        if mode == "DATA":
            assert not file == None
            
        obs_file = None
        if not file == None:
            obs_file = open(file, 'a', encoding='utf-8')
            
        if mode == "ANN":
            assert not self.model == None
            
        
        #Setup PyGame environment
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode([SCREEN_WIDTH,SCREEN_HEIGHT])
    
        direction = 'RIGHT'
        change_dir = direction
        
        while self.running:
            
            angle = math.degrees(math.atan2(self.food.rect.x - self.player.anchorfront[0]
                                            , self.food.rect.y - self.player.anchorfront[1]))/180
            distance = math.sqrt((self.player.rect.x - self.food.rect.x)**2 + 
                                 (self.player.rect.y - self.food.rect.y)**2)
            x_dif = abs((self.player.rect.x - self.food.rect.x)/SCREEN_WIDTH)
            y_dif = abs((self.food.rect.y - self.player.rect.y)/SCREEN_HEIGHT)
            
            #Event Handler
            for event in pygame.event.get():
                if event.type == pygame.QUIT: #The window is closed
                    self.running = False
        
                #Key press event
                if event.type == KEYDOWN: 
                    if event.key == K_ESCAPE: #If escape pressed, exit run state
                        self.running = False
                    elif event.key == K_UP:
                        change_dir = 'UP'
                    elif event.key == K_DOWN:
                        change_dir = 'DOWN'
                    elif event.key == K_RIGHT:
                        change_dir = 'RIGHT'
                    elif event.key == K_LEFT:
                        change_dir = 'LEFT'
            
            #Determining whether one of the collision blocks overlaps with an
            #obstacle
            front = 0
            left = 0
            right = 0
            
            if pygame.sprite.spritecollide(self.player.col_blockF,
                                           self.walls, False):
                front = 1
            if pygame.sprite.spritecollide(self.player.col_blockL,
                                           self.walls, False):
                left = 1
            if pygame.sprite.spritecollide(self.player.col_blockR,
                                           self.walls, False):
                right = 1
            
            for i in self.player.body:
                if pygame.sprite.collide_rect(self.player.col_blockF, i):
                    front = 1
                if pygame.sprite.collide_rect(self.player.col_blockL, i):
                    left = 1
                if pygame.sprite.collide_rect(self.player.col_blockR, i):
                    right = 1
            
            #Recording for observation
            sug_move = None 
            if mode == "ANN":
                change_dir = self.network_move(front, left, right, angle, x_dif,
                                          y_dif, direction, change_dir,
                                          self.player, self.food)
            elif mode == "DATA":    
                change_dir = self.random_move(direction, change_dir)
                sug_move = self.observation_suggested_move(direction, change_dir)
                
            #else: mode set to "HUMAN"
            
            #Movement validation to prevent backwards movement
            if change_dir == 'UP' and direction != 'DOWN':
                direction = 'UP'
            elif change_dir == 'DOWN' and direction != 'UP':
                direction = 'DOWN'
            elif change_dir == 'LEFT' and direction != 'RIGHT':
                direction = 'LEFT'
            elif change_dir == 'RIGHT' and direction != 'LEFT':
                direction = 'RIGHT'
                
            #Create a body segment of the heads position before movement
            b = Body(self.player.rect.x+12.5, self.player.rect.y+12.5) 
            
            self.player.update(direction, change_dir)
            
            new_x_dif = abs((self.player.rect.x - self.food.rect.x)/SCREEN_WIDTH)
            new_y_dif = abs((self.food.rect.y - self.player.rect.y)/SCREEN_HEIGHT)
        
            #new_player_loc = (self.player.rect.x, self.player.rect.y)        
            new_distance = math.sqrt((self.player.rect.x - self.food.rect.x)**2
                                     + (self.player.rect.y - self.food.rect.y)**2)
            
            #Screen Drawing
            self.screen_drawing(screen)
            
            if additional_params["GRID"]:
            #Draw gridlines
                for i in np.arange(0, 525, 25):
                    pygame.draw.aaline(screen, RED, (i, 0), (i, SCREEN_HEIGHT))
                    pygame.draw.aaline(screen, RED, (0, i), (SCREEN_WIDTH, i))
            
            #Fruit Collision
            self.player.body.insert(0, b) #For every variable popped from the body, the head is readded at its new position
            if pygame.sprite.spritecollide(self.player, self.all_sprites, False):
                self.score_value+=1
                self.food.update()
            else:
                self.player.body.pop() 
                #The snake is being continually drawn, this removes the last item from the snake body and gives the illusion that it is actually travelling
            
            for unit in self.player.body:
                screen.blit(unit.surf,unit.rect)
            
            for i in self.player.body:
                if pygame.sprite.collide_rect(self.player, i):
                    self.snake_die = True
            
            if pygame.sprite.spritecollide(self.player, self.walls, False):
                self.snake_die = True
            
            correct_move = None
            if self.snake_die:
                correct_move = -1
            elif distance-new_distance <= 0:
                correct_move = 0 #Wrong move, distance got larger
            elif distance-new_distance > 0:
                correct_move = 1 #Correct move, distance got smaller
            
            if mode == "DATA":
                assert not obs_file == None
                
                observation = str(front) + ',' + str(left) + ',' + str(right) 
                + ',' + f'{angle:.2f}' + ',' + str(x_dif) + ',' + str(y_dif) 
                + ',' + str(new_x_dif) + ',' + str(new_y_dif)+ ',' 
                + str(sug_move) + ',' + str(correct_move) + '\n'
                obs_file.write(observation)
            
            if self.snake_die:
                self.running = False
            
            #Game system handling
            clock.tick(additional_params["FPS"])
            pygame.display.set_caption('Score: {} Front: {} Left: {} Right: {} Angle {:.2f} Distance {:.2f}'.format(str(self.score_value), str(front), str(left), str(right), angle, distance))
            pygame.display.update() #Updates the display with the events that have occured since the last flip call
        
        #After game cleanup
        pygame.quit()
        if not file == None:
            obs_file.close()
            
    def screen_drawing(self, screen):
        screen.fill(BLACK) #Draw black on the screen
        screen.blit(self.food.surf, self.food.rect)
        screen.blit(self.player.surf, self.player.rect)
        #Draw Wall
        screen.blit(self.wallL.surf, self.wallL.rect)
        screen.blit(self.wallU.surf, self.wallU.rect)
        screen.blit(self.wallR.surf, self.wallR.rect)
        screen.blit(self.wallD.surf, self.wallD.rect)
        #Draw collision 
        screen.blit(self.player.col_blockF.surf, self.player.col_blockF.rect)
        screen.blit(self.player.col_blockL.surf, self.player.col_blockL.rect)
        screen.blit(self.player.col_blockR.surf, self.player.col_blockR.rect)
        
    def observation_suggested_move(self, direct, change_direct):
        '''
        Records the move made based on current travel and the input
        direction.
        
        0 corresponds to forward input
        1 corresponds to right input
        -1 corresponds to left input
        '''
        
        if direct == 'RIGHT':
            if change_direct == 'RIGHT':
                return 0
            elif change_direct == 'UP':
                return -1
            elif change_direct == 'DOWN':
                return 1
            else:
                return 0
        elif direct == 'LEFT':
            if change_direct == 'LEFT':
                return 0
            elif change_direct == 'UP':
                return 1
            elif change_direct == 'DOWN':
                return -1
            else:
                return 0
        elif direct == 'UP':
            if change_direct == 'UP':
                return 0
            elif change_direct == 'LEFT':
                return -1
            elif change_direct == 'RIGHT':
                return 1
            else:
                return 0
        elif direct == 'DOWN':
            if change_direct == 'DOWN':
                return 0
            elif change_direct == 'LEFT':
                return 1
            elif change_direct == 'RIGHT':
                return -1
            else:
                return 0
            
    def network_move(self, front, left, right, angle, x_dif, y_dif, direct,
                     change_dir, player, food):
        if direct == 'RIGHT':
            #dif for left move
            y_new_l = abs((player.rect.y -25) - food.rect.y)/SCREEN_HEIGHT
            x_new_l = x_dif
            #dif for forward move
            y_new_f = y_dif
            x_new_f = abs((x_dif + 25) - food.rect.x)/SCREEN_WIDTH
            #dif for right move
            y_new_r = abs((player.rect.y +25) - food.rect.y)/SCREEN_HEIGHT
            x_new_r = x_dif
            
        elif direct == 'LEFT':
            #dif for right move
            y_new_r = abs((player.rect.y -25) - food.rect.y)/SCREEN_HEIGHT
            x_new_r = x_dif
            #dif for forward move
            y_new_f = y_dif
            x_new_f = abs((x_dif - 25) - food.rect.x)/SCREEN_WIDTH
            #dif for left move
            y_new_l = abs((player.rect.y +25) - food.rect.y)/SCREEN_HEIGHT
            x_new_l = x_dif
            
        elif direct == 'UP':
            #dif for left move
            y_new_l = y_dif
            x_new_l = abs((player.rect.x - 25) - food.rect.x)/SCREEN_WIDTH
            #dif for forward move
            y_new_f = abs((player.rect.y -25) - food.rect.y)/SCREEN_HEIGHT
            x_new_f = x_dif
            #dif for right move
            y_new_r = y_dif
            x_new_r = abs((player.rect.x + 25) - food.rect.x)/SCREEN_WIDTH
        
        elif direct == 'DOWN':
            #dif for left move
            y_new_l = y_dif
            x_new_l = abs((player.rect.x + 25) - food.rect.x)/SCREEN_WIDTH
            #dif for forward move
            y_new_f = abs((player.rect.y + 25) - food.rect.y)/SCREEN_HEIGHT
            x_new_f = x_dif
            #dif for right move
            y_new_r = y_dif
            x_new_r = abs((player.rect.x - 25) - food.rect.x)/SCREEN_WIDTH
            
        #Create three arrays to hold each suggested move, and check to see which is best
        choice1 = np.array([front, left, right, angle, x_dif, y_dif, x_new_l, y_new_l, -1])[np.newaxis, :]
        choice2 = np.array([front, left, right, angle, x_dif, y_dif, x_new_f, y_new_f, 0])[np.newaxis, :]
        choice3 = np.array([front, left, right, angle, x_dif, y_dif, x_new_r, y_new_r, 1])[np.newaxis, :]
        
        predictions = [self.model.predict(choice1)[0][1], self.model.predict(choice2)[0][1], self.model.predict(choice3)[0][1]]
        index_max = max(range(len(predictions)), key=predictions.__getitem__)
        
        if index_max == 0:
            move = -1
        elif index_max == 1:
            move = 0
        else:
            move = 1
    
        if direct == 'RIGHT':
            if move == -1: #Left
                change_dir = 'UP'
            elif move == 1: #Right
                change_dir = 'DOWN'
            else: 
                change_dir = direct
        if direct == 'LEFT':
            if move == -1: #Left
                change_dir = 'DOWN'
            elif move == 1: #Right
                change_dir = 'UP'
            else: #Forward
                change_dir = direct
        if direct == 'UP':
            if move == -1: #Left
                change_dir = 'LEFT'
            elif move == 1: #Right
                change_dir = 'RIGHT'
            else: #Forward
                change_dir = direct
        if direct == 'DOWN':
            if move == -1: #Left
                change_dir = 'RIGHT'
            elif move == 1: #Right
                change_dir = 'LEFT'
            else: #Forward
                change_dir = direct
        return change_dir