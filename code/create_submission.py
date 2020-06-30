def agent_function(observation, configuration):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import math
    import base64
    import io

    N = 7
    M = 6
    WIN = 10000
    LOSS = -10000

    def validMove(board, move):
        return board[move] == 0

    def makeMove(board, move, player):
        newBoard = list(board)
        # drop it
        if validMove(newBoard, move) is False:
            return -1

        last = 0
        for i in range(move, N * M, N):
            if newBoard[i] == 0:
                last = i
                continue
            break

        newBoard[last] = player
        return newBoard

    def points(board, player):
        result = 0

        for i in range(len(board)):
            n = i % N
            m = math.floor(i / N)

            if board[i] == player:
                # horizontal
                k = 0
                for j in range(i + 1, (m + 1) * N):
                    k = k + 1
                    if board[j] == player:
                        result = result + k
                        if k == 3:
                            return WIN
                    else:
                        break

                # vertical
                k = 0
                for j in range(i + N, M * N, N):
                    k = k + 1
                    if board[j] == player:
                        result = result + k
                        if k == 3:
                            return WIN
                    else:
                        break

                # diagonal right
                k = 0
                for j in range(1, N - n):
                    k = k + 1
                    next = i + ((N + 1) * j)
                    if next < (N * M) and board[next] == player:
                        result = result + k
                        if k == 3:
                            return WIN
                    else:
                        break

                # diagonal left
                k = 0
                for j in range(1, n + 1):
                    k = k + 1
                    next = i + ((N - 1) * j)
                    if next < (N * M) and board[next] == player:
                        result = result + k
                        if k == 3:
                            return WIN
                    else:
                        break

        return result

    def will_lose(board, action, player_id):
        next_board = makeMove(board, action, player_id)
        other_player = 1 if player_id == 2 else 2
        if next_board != -1 and points(next_board, player_id) == WIN:
            return False

        if next_board == -1:
            return False

        for j in range(7):
            next_board_other = makeMove(next_board, j, other_player)
            if next_board_other == -1:
                continue
            if points(next_board_other, other_player) == WIN:
                return True
        return False

    def can_win(board, player_id):
        for i in range(7):
            next_board = makeMove(board, i, player_id)
            if next_board != -1 and points(next_board, player_id) == WIN:
                return i
        return -1

    class Network(nn.Module):

        def __init__(self, rows, columns, in_channels=1):
            super().__init__()
            self.rows = rows
            self.columns = columns
            self.in_channels = in_channels
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
            # self.batch_norm1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
            # self.batch_norm2 = nn.BatchNorm2d(16)
            self.fc1 = nn.Linear(in_features=16 * 6 * 7, out_features=128)
            self.out = nn.Linear(in_features=128, out_features=columns)

        def forward(self, t):
            # (2) hidden conv layer
            t = self.conv1(t)
            # t = self.batch_norm1(t)
            t = F.relu(t)

            # (3) hidden conv layer
            t = self.conv2(t)
            # t = self.batch_norm2(t)
            t = F.relu(t)

            # (4) hidden linear layer
            t = torch.flatten(t, start_dim=1)
            t = self.fc1(t)
            t = F.relu(t)

            # (5) output layer
            t = self.out(t)

            return t

    device = torch.device('cpu')
    policy_net = Network(6, 7)
    encoded_weights = """
    BASE64_PARAMS"""
    decoded = base64.b64decode(encoded_weights)
    buffer = io.BytesIO(decoded)
    policy_net.load_state_dict(torch.load(buffer, map_location=device))
    policy_net.eval()

    def preprocess(board, mark):
        return np.array([1 if val == mark else 0 if val == 0 else 2 for val in board])

    def get_preprocessed_state():
        board = observation.board
        mark = observation.mark
        board_state = preprocess(board, mark)
        return torch.tensor(board_state, dtype=torch.float32).reshape(1, 1, 6, 7)

    state = get_preprocessed_state()

    board = np.array(observation.board).reshape(6, 7)
    valid_actions = []

    with torch.no_grad():
        prediction = policy_net(state).argmax(dim=1).item()

    win_action = can_win(observation.board, observation.mark)

    desired_actions = []

    for i in range(7):
        if np.count_nonzero(board[:, i]) != 6:
            valid_actions.append(i)

        if np.count_nonzero(board[:, i]) != 6 and not will_lose(observation.board, i, observation.mark):
            desired_actions.append(i)

    if win_action != -1:
        return win_action

    if prediction in desired_actions:
        return int(prediction)
    else:
        if len(desired_actions) > 0:
            return desired_actions[0]
        return int(np.random.choice(valid_actions))

import inspect
import os
import base64
import sys
from constants import Hyperparameters
from kaggle_environments import utils, make

no_params_path = "submission_template.py"

def append_object_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)


def write_agent_to_file(function, file):
    with open(file, "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)


write_agent_to_file(agent_function, no_params_path)


model_id = 2000
INPUT_PATH = f'{Hyperparameters.model_base_path}{Hyperparameters.model_base_name}{model_id}'
OUTPUT_PATH = f'agents/agent_{Hyperparameters.model_base_name}{model_id}.py'

with open(INPUT_PATH, 'rb') as f:
    raw_bytes = f.read()
    encoded_weights = base64.encodebytes(raw_bytes).decode()

with open(no_params_path, 'r') as file:
    data = file.read()

data = data.replace('BASE64_PARAMS', encoded_weights)

with open(OUTPUT_PATH, 'w') as f:
    f.write(data)
    print('written agent with net parameters to', OUTPUT_PATH)

print()
print('Testing the created agent')

out = sys.stdout
try:
    submission = utils.read_file(OUTPUT_PATH)
    agent = utils.get_last_callable(submission)
finally:
    sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")

