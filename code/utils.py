import numpy as np
import torch

def preprocess(board, mark):
    return np.array([1 if val == mark else 0 if val == 0 else 2 for val in board])

def choose_action(state, q_net):
    board = torch.flatten(state, start_dim=0).numpy().reshape(6, 7)
    valid_actions = []
    for i in range(7):
        if np.count_nonzero(board[:, i]) != 6:
            valid_actions.append(i)
    prediction = q_net(state).argmax(dim=1).item()
    if prediction in valid_actions:
        return int(prediction)
    else:
        return int(np.random.choice(valid_actions))

def create_symetry(s):
    s = s.reshape(6, 7)
    s_ = torch.zeros(42).reshape(6, 7)
    for i in range(7):
        s_[:, i] = s[:, 6 - i]
    return s_.reshape(1, 6, 7)

def create_symetry_memory(state, action, next_state, reward):
    s_ = create_symetry(state)
    ns_ = create_symetry(next_state)
    a_ = 6 - action
    return s_, a_, ns_, reward

def create_dual_memory(memory):
    return [create_symetry_memory(s, a, ns, r) for s, a, ns, r in memory]

def mean_reward(rewards):
    return sum(r[0] if r[0] == 1 else 0 for r in rewards) / len(rewards)
