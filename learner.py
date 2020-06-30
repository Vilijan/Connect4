import numpy as np

import torch

import torch.optim as optim
import torch.nn.functional as F

from models import QNetwork
from exploration_strategies import EpsilonGreedyStrategy
from constants import Hyperparameters


class Connect4Learner:
    def __init__(self,
                 rows,
                 columns,
                 memory,
                 learning_rate):
        self.policy_net = QNetwork(rows, columns)
        self.target_net = QNetwork(rows, columns)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = memory
        self.actions = columns
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.loss = F.mse_loss
        self.strategy = EpsilonGreedyStrategy(Hyperparameters.EPS_START,
                                              Hyperparameters.EPS_END,
                                              Hyperparameters.EPS_DECAY)
        self.total_turns = 0
        self.avg_loss_per_fit = []

    def dqn_train(self, batch):
        state, action, next_state, reward = batch

        reward = reward.squeeze(1)
        action = action.long()

        q = self.policy_net(state).gather(dim=1, index=action).squeeze(1)
        q_, _ = self.target_net(next_state).max(dim=1)

        sums = torch.sum(next_state.abs(), dim=[1, 2, 3])
        q_[sums == 0] = 0

        predicted = reward - q_ * Hyperparameters.GAMMA
        self.optimizer.zero_grad()

        curr_loss = self.loss(predicted, q)
        curr_loss.backward()
        self.optimizer.step()

        return curr_loss.item()

    def ddqn_train(self, batch):
        state, action, next_state, reward = batch

        reward = reward.squeeze(1)
        action = action.long()

        q = self.policy_net(state).gather(dim=1, index=action).squeeze(1)
        q_next = self.policy_net(next_state).argmax(dim=1).unsqueeze(1)

        q_ = self.target_net(next_state).gather(dim=1, index=q_next).squeeze(1)
        sums = torch.sum(next_state.abs(), dim=[1, 2, 3])

        q_[sums == 0] = 0
        predicted = reward - q_ * Hyperparameters.GAMMA

        self.optimizer.zero_grad()
        curr_loss = self.loss(predicted, q)
        curr_loss.backward()
        self.optimizer.step()

        return curr_loss.item()

    def fit(self, epochs):
        if len(self.memory.memory) < Hyperparameters.MIN_MEMORY_SIZE:
            return

        dataloader, dataloader_dual = self.memory.dataloader(Hyperparameters.BATCH_SIZE)

        final_loss = 0.0
        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            if epoch % 2 == 0:
                dl = dataloader
            else:
                dl = dataloader_dual
            for batch_ndx, sample in enumerate(dl):
                total_loss += self.ddqn_train(sample)
                batches += 1
            final_loss += total_loss
            print("Average loss per epoch", total_loss)
        final_loss /= epochs
        self.avg_loss_per_fit.append(final_loss)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_optimizer(self, learning_rate):
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

    def get_action(self, state):
        self.total_turns += 1
        epsilon = self.strategy.get_exploration_rate(self.total_turns)
        board = torch.flatten(state, start_dim=0).numpy().reshape(6, 7)
        valid_actions = []

        for i in range(7):
            if np.count_nonzero(board[:, i]) != 6:
                valid_actions.append(i)

        if np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))
        else:
            with torch.no_grad():
                prediction = self.policy_net(state).argmax(dim=1)
            return prediction.item()