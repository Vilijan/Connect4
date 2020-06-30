import torch
from utils import choose_action

def play_episode(env,
                 learner,
                 opponent=None,
                 my_agent_id=None):
    env.reset()
    my_agent_won = True

    while not env.game_over():
        active = env.current_player()
        state = torch.tensor(env.get_preprocessed_state(), dtype=torch.float32)

        if my_agent_id is None or active == my_agent_id:
            action = learner.get_action(state.unsqueeze(0))
        else:
            action = choose_action(state.unsqueeze(0), opponent)

        env.step([action if i == active else None for i in [0, 1]])
        reward = env.get_state()[active].reward

        if env.game_over():
            if reward == 1:  # Won
                if my_agent_id != active:
                    my_agent_won = False
                elif reward == 0:  # Lost
                    if active == my_agent_id:
                        my_agent_won = False
                    reward = -1
                else:
                    reward = 0
            else:
                reward = 0

        if reward != 0 and my_agent_id is not None:
            reward = 1 if my_agent_won else -1

        if not env.valid_action(action):
            reward = -1

        next_state = torch.tensor(env.get_preprocessed_state(), dtype=torch.float32)

        if env.game_over():
            next_state = torch.zeros(42).reshape(1, 6, 7)

        reward = torch.tensor([reward], dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.float32)
        learner.memory.add(state, action, next_state, reward)

    return my_agent_won
