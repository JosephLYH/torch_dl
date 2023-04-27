import copy
import os
import random
import torch

class BattleshipEnv:
    def __init__(self, players):
        # constants
        self.multiplayer = players == 2
        self.ship_lengths = [2, 3, 3, 4, 5]
        self.ship_names = ['Destroyer', 'Submarine', 'Cruiser', 'Battleship', 'Carrier']
        self.action_dim = (10, 10)
        self.state_dim = (2, *self.action_dim)
        self.guess_map = {
            0: ' ',
            1: '.',
            2: 'X',
        }
        self.ship_map = {
            0: ' ',
            1: 'O',
        }
        self.reset()

    def reset(self):
        # states
        self.game_over = False
        self.winner = None
        self.player_turn = 0
        self.turns = 0
        
        self.grid = [torch.zeros(self.state_dim) for _ in range(2)]
        self.occupied = [{
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
        } for _ in range(2)]

    def render(self, player=None):
        if player != 1:
            print("Player 0's state:")
            for y in self.grid[0][1].tolist(): print('|' + '|'.join([self.guess_map[x] for x in y]) + '|')
            print("Player 1's board:")
            for y in self.grid[1][0].tolist(): print('|' + '|'.join([self.ship_map[x] for x in y]) + '|')
            print()

        if player != 0:
            print("Player 1's state:")
            for y in self.grid[1][1].tolist(): print('|' + '|'.join([self.guess_map[x] for x in y]) + '|')
            print("Player 0's board:")
            for y in self.grid[0][0].tolist(): print('|' + '|'.join([self.ship_map[x] for x in y]) + '|')
            print()

    def random_init(self):
        self.reset()
        for player in range(2):
            for ship in range(5):
                while True:
                    x = random.randint(0, 9)
                    y = random.randint(0, 9)
                    direction = random.randint(0, 1)
                    if self.check_valid_init(player, x, y, direction, self.ship_lengths[ship]):
                        self.place_ship(player, x, y, direction, ship)
                        break

        return copy.deepcopy(self.grid[0][1])

    def check_valid_init(self, player, x, y, direction, length):
        if direction == 0:
            if x + length > 10:
                return False
            for i in range(length):
                if self.grid[player][0][y][x+i] != 0:
                    return False
        else:
            if y + length > 10: 
                return False
            for i in range(length):
                if self.grid[player][0][y+i][x] != 0:
                    return False
        return True
    
    def place_ship(self, player, x, y, direction, ship):
        if direction == 0:
            for i in range(self.ship_lengths[ship]):
                self.grid[player][0][y][x+i] = 1
                self.occupied[player][ship].append([x+i, y])
        else:
            for i in range(self.ship_lengths[ship]):
                self.grid[player][0][y+i][x] = 1
                self.occupied[player][ship].append([x, y+i])

    def step(self, player, action):
        reward = -1
        self.turns += 1
        x, y = action

        # wait for player's turn
        while not self.multiplayer and self.player_turn != player:
            pass
        
        # check if action is not valid
        if self.grid[player][1][y][x] != 0:
            self.turns -= 1
            reward = -100
            return copy.deepcopy(self.grid[player][1]), reward, self.game_over
        
        # check if action is a hit
        if self.grid[1 - player][0][y][x] != 0:
            reward = 1
            self.grid[player][1][y][x] = 2

            for ship in list(self.occupied[1 - player].keys()):       
                # check which ship was hit
                if action.tolist() in self.occupied[1 - player][ship]:
                    self.occupied[1 - player][ship].remove(action.tolist())

                    # check if ship is sunk
                    if len(self.occupied[1 - player][ship]) == 0:
                        del self.occupied[1 - player][ship]

                        # check if game is over
                        if len(self.occupied[1 - player]) == 0:
                            self.game_over = True
                            self.winner = player
                            reward = 10
                            
                            return copy.deepcopy(self.grid[player][1]), reward, self.game_over
        else:
            self.grid[player][1][y][x] = 1
        
        next_state = copy.deepcopy(self.grid[player][1])

        if self.multiplayer: 
            self.change_turn()

        return next_state, reward, self.game_over
    
    def change_turn(self):
        self.player_turn = 1 - self.player_turn

    def get_next_turn_state(self):
        return copy.deepcopy(self.grid[self.player_turn][1])
    