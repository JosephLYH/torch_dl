import copy
import os
import random
import time
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib.agents import Agent
from lib.nets import BattleshipNet
from lib.utils import write_tb

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class BattleshipEnv:
    def __init__(self):
        # constants
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
        while self.player_turn != player:
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

        self.player_turn = 1 - self.player_turn
        return next_state, reward, self.game_over

    def get_next_turn_state(self):
        return copy.deepcopy(self.grid[self.player_turn][1])

class BattleshipAgent(Agent):
    def __init__(self, player, state_dim, action_dim):
        super().__init__()

        self.device = 'cpu'
        self.player = player
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory = deque(maxlen=8192)
        self.batch_size = 512
        self.gamma = 0.9
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 512  # no. of experiences between saving Net

        # DNN to predict the most optimal action - we implement this in the Learn section
        self.net = BattleshipNet()
        self.net = self.net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4, eps=1e-7)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = self.batch_size  # min. experiences before training
        self.learn_every = 25  # no. of experiences between updates to Q_online
        self.sync_every = 25  # no. of experiences between Q_target & Q_online sync

        self.time_start = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.save_dir = os.path.join('battleship', self.time_start, 'checkpoints', str(self.player))
        self.scaler = torch.cuda.amp.GradScaler(growth_interval=100)
        self.scaler._init_scale = 2.**10
        self.writer = SummaryWriter(log_dir=os.path.join('battleship', self.time_start, 'tb', str(self.player)))

    def act(self, state):
        state = state.to(self.device)

        if np.random.rand() < self.exploration_rate: # EXPLORE
            y, x = np.unravel_index(torch.argmax(torch.rand(*self.action_dim)).cpu(), self.action_dim)
        else: # EXPLOIT
            action_values = self.net(state.unsqueeze(0).unsqueeze(0), model='online')
            y, x = np.unravel_index(torch.argmax(action_values.squeeze(0).squeeze(0)).cpu(), self.action_dim)
        
        action_idx = torch.tensor([x, y])

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = list(zip(*batch))

        state = torch.stack(state).unsqueeze(1)
        next_state = torch.stack(next_state).unsqueeze(1)
        action = torch.stack(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done)     
        return state, next_state, action, reward, done

    def td_estimate(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        
        action_x, action_y = torch.tensor(list(zip(*action)))
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), 0, action_x, action_y]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        next_state_Q = self.net(next_state, model='online')
        best_action_flatten = torch.argmax(next_state_Q.squeeze(1).view(self.batch_size, -1), -1)
        best_action = torch.stack([best_action_flatten // self.action_dim[1], best_action_flatten % self.action_dim[1]], -1)
        best_action_x, best_action_y = torch.tensor(list(zip(*best_action)))
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), 0, best_action_x, best_action_y]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        td_estimate = td_estimate.to(self.device)
        td_target = td_target.to(self.device)

        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        write_tb(self.curr_step, self.writer, self.net, self.scaler)
        save_path = os.path.join(self.save_dir, f'{self.curr_step:020d}.pt')
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f'Net {self.player} saved to {save_path} at step {self.curr_step}')


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
    
    def to(self, device):
        self.device = device
        self.net = self.net.to(device)
    
if __name__ == '__main__':
    env = BattleshipEnv()
    agents = [BattleshipAgent(i, env.action_dim, env.action_dim) for i in range(2)]

    episodes = 100000
    for e in range(episodes):
        state = env.random_init()

        # Play the game!
        step = 0
        while True:
            if step % 50 == 0:
                print(f'episode: {e}, step: {step}, agent {env.player_turn}')
                print(f'exploration rate: {agents[env.player_turn].exploration_rate}')
                env.render(env.player_turn)

            agents[env.player_turn].to('cuda')
            action = agents[env.player_turn].act(state)
            next_state, reward, done = env.step(env.player_turn, action)
            agents[env.player_turn].cache(state, next_state, action, reward, done)
            q, loss = agents[env.player_turn].learn()
            agents[env.player_turn].to('cpu')

            if done: break

            state = env.get_next_turn_state()
            step += 1