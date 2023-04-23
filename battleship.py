import copy
import datetime
import os
import random
import time
from collections import deque
from pathlib import Path

import gym
import gym_super_mario_bros
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.spaces import Box
from gym.wrappers import FrameStack
from PIL import Image
from torch import nn
from torchvision import transforms as T

from lib import nets

class BattleshipEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # constants
        self.ship_lengths = [2, 3, 3, 4, 5]
        self.ship_names = ['Destroyer', 'Submarine', 'Cruiser', 'Battleship', 'Carrier']
        
        # states
        self.game_over = False
        self.winner = None
        self.player_turn = 0
        self.turns = 0
        
        self.ship_sunk = [0, 0]
        self.grid = [torch.zeros((2, 10, 10)) for _ in range(2)]
        self.occupied = [{
            'all': set(),
            0: set(),
            1: set(),
            2: set(),
            3: set(),
            4: set(),
            5: set(),
        } for _ in range(2)]

    def render(self):
        print("Player 1's board:")
        print(self.grid[0][0])
        print("Player 1's guesses:")
        print(self.grid[0][1])
        print()
        print("Player 2's board:")
        print(self.grid[1][0])
        print("Player 2's guesses:")
        print(self.grid[1][1])

    def random_init(self):
        self.reset()
        for player in range(2):
            for ship in range(5):
                while True:
                    x = random.randint(0, 9)
                    y = random.randint(0, 9)
                    direction = random.randint(0, 1)
                    if self.check_valid(player, x, y, direction, self.ship_lengths[ship]):
                        self.place_ship(player, x, y, direction, ship)
                        break

    def check_valid(self, player, x, y, direction, length):
        if direction == 0:
            if x + length > 10: 
                return False
            for i in range(length):
                if (x+i, y) in self.occupied[player]['all']:
                    return False
        else:
            if y + length > 10: 
                return False
            for i in range(length):
                if (x, y+i) in self.occupied[player]['all']:
                    return False

        return True
    
    def place_ship(self, player, x, y, direction, ship):
        if direction == 0:
            for i in range(self.ship_lengths[ship]):
                self.grid[player][0][y][x+i] = 1
                self.occupied[player]['all'].add((x+i, y))
                self.occupied[player][ship].add((x+i, y))
        else:
            for i in range(self.ship_lengths[ship]):
                self.grid[player][0][y+i][x] = 1
                self.occupied[player]['all'].add((x, y+i))
                self.occupied[player][ship].add((x, y+i))

    def check_sunk(self, player, x, y):
        for ship in range(5):
            if (x, y) in self.occupied[1-player][ship]:
                self.occupied[1-player][ship].remove((x, y))
                if len(self.occupied[1-player][ship]) == 0:
                    self.ship_sunk[1-player] += 1
                    return True
        return False

    def step(self, player, action):
        reward = -1
        self.turns += 1
        x, y = action
        
        if self.grid[player][1][y][x] != 0:
            return self.grid[player], reward, self.game_over
        
        if action in self.occupied[1 - player]['all']:
            reward = 1
            self.grid[player][1][y][x] = 2
            for ship in range(5):
                

        else:
            reward = 0
            self.grid[player][1][y][x] = 1

        next_state = copy.deepcopy(self.grid[player])

        return next_state, reward, self.game_over

if __name__ == '__main__':
    env = BattleshipEnv()
    env.random_init()
    env.step(0, (0, 0))
    env.render()