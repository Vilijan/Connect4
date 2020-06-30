from models import QNetwork
import torch
from constants import Hyperparameters
import numpy as np
from utils import choose_action
from kaggle_environments import evaluate
from utils import mean_reward

model_id = 2000
agent = QNetwork(6, 7)
agent.load_state_dict(torch.load(f'{Hyperparameters.model_base_path}{Hyperparameters.model_base_name}{model_id}'))
agent.eval()

def my_agent(observation, configuration):
    global agent
    state = observation
    state = np.array([1 if val == state.mark else 0 if val == 0 else 2 for val in state.board])
    state = torch.tensor(state, dtype=torch.float32).reshape(1, 1, 6, 7)
    action = choose_action(state, agent)
    return action

print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
print("Random Agent vs My Agent", mean_reward(evaluate("connectx", ["random", my_agent], num_episodes=10)))