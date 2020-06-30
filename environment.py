import gym
from kaggle_environments import make
import torch
from utils import preprocess

class ConnectX(gym.Env):
    def __init__(self):
        self.env = make('connectx', debug=False)

        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)

    def valid_action(self, action):
        return self.env.state[0].observation.board[action] == 0

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def get_state(self):
        return self.env.state

    def get_preprocessed_state(self):
        board = self.env.state[0].observation.board
        mark = self.env.state[self.current_player()].observation.mark
        board_state = preprocess(board, mark)
        return torch.tensor(board_state, dtype=torch.float32).reshape(1, 6, 7)

    def game_over(self):
        return self.env.done

    def current_player(self):
        active = -1
        if self.env.state[0].status == "ACTIVE":
            active = 0
        if self.env.state[1].status == "ACTIVE":
            active = 1
        return active

    def get_configuration(self):
        return self.env.configuration