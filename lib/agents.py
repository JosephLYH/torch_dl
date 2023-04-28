import os
import random
import time
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from lib.nets import BattleshipNet, MarioNet
from lib.utils import write_tb
from lib.optimizers.lion import Lion


class Agent():
    def __init__(self):
        """Init"""

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""

    def cache(self, experience):
        """Add the experience to memory"""

    def recall(self):
        """Sample experiences from memory"""

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""

class Mario(Agent):
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.9
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5e5  # no. of experiences between saving Mario Net

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=2.5e-4)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
    Outputs:
    ``action_idx`` (``int``): An integer representing which action Mario will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

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
    
class Battleship(Agent):
    def __init__(self, player, state_dim, action_dim, episodes=1000000):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.player = player
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episodes = episodes

        self.memory = deque(maxlen=32768)
        self.batch_size = 1024
        self.gamma = 0.9
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5000  # no. of experiences between saving Net

        # DNN to predict the most optimal action - we implement this in the Learn section
        self.enable_f16 = True
        self.net = BattleshipNet()
        self.net = self.net.cuda()
        self.optimizer = Lion(self.net.parameters(), lr=1, weight_decay=1e-2)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: max(1e-4 - (1e-4 - 1e-6)*step/(episodes*50), 1e-6))
        self.scaler = torch.cuda.amp.GradScaler(growth_interval=100, enabled=self.enable_f16)
        self.scaler._init_scale = 2.**8

        self.burnin = self.batch_size  # min. experiences before training
        self.learn_every = 64  # no. of experiences between updates to Q_online
        self.sync_every = 5000  # no. of experiences between Q_target & Q_online sync

        self.time_start = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.save_dir = os.path.join('battleship', self.time_start, 'checkpoints', str(self.player))
        self.writer = SummaryWriter(log_dir=os.path.join('battleship', self.time_start, 'tb', str(self.player)))

    def act(self, state):
        state = state.to(self.device)

        if np.random.rand() < self.exploration_rate: # EXPLORE
            y, x = np.unravel_index(torch.argmax(torch.where(state == 0, torch.rand(*self.action_dim).to(self.device), torch.tensor(-1e2).to(self.device))).cpu(), self.action_dim) # random action from legal actions
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
            current_Q = self.net(state, model='online')
        current_Q = torch.where(state == 0, current_Q, torch.tensor(-1e2).cuda()) # mask out illegal moves
        current_Q = current_Q[np.arange(0, self.batch_size), 0, action_x, action_y] # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.enable_f16):
            next_state_Q = self.net(next_state, model='online')
        next_state_Q = next_state_Q.float()
        next_state_Q = torch.where(next_state == 0, next_state_Q, torch.tensor(-1e2).cuda()) # mask out illegal moves
        best_action_flatten = torch.argmax(next_state_Q.squeeze(1).view(self.batch_size, -1), -1)
        best_action = torch.stack([best_action_flatten // self.action_dim[1], best_action_flatten % self.action_dim[1]], -1)
        best_action_x, best_action_y = torch.tensor(list(zip(*best_action)))

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.enable_f16):
            next_Q = self.net(next_state, model='target')
        next_Q = next_Q.float()
        next_Q = torch.where(next_state == 0, next_Q, torch.tensor(-1e2).cuda()) # mask out illegal moves
        next_Q = next_Q[np.arange(0, self.batch_size), 0, best_action_x, best_action_y]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        td_estimate = td_estimate.to(self.device)
        td_target = td_target.to(self.device)
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = os.path.join(self.save_dir, f'{self.curr_step:020d}.pt')
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(
            {
                'model': self.net.state_dict(), 
                'optimizer': self.optimizer.state_dict(),
                'scaler': self.scaler.state_dict(),
                'curr_step': self.curr_step,
                'exploration_rate': self.exploration_rate,
                'memory': self.memory,
            },
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
    
    def write_tb(self, metrics):
        write_tb(self.curr_step, self.writer, metrics, self.net, self.optimizer, self.scaler)
    
    def to(self, device):
        self.device = device
        self.net = self.net.to(device)