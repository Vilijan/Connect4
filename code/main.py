from environment import ConnectX
from experience_memory import ReplayMemory
from constants import Hyperparameters
from learner import Connect4Learner
import torch
from selfplay import play_episode

env = ConnectX()

available_actions = env.action_space.n

replay_memory = ReplayMemory(Hyperparameters.MEMORY_SIZE)

input_features = env.observation_space.n

learner = Connect4Learner(input_features, available_actions, replay_memory, Hyperparameters.LEARNING_RATE)

episode = 1

while episode < Hyperparameters.EPISODES:
    episode += 1
    play_episode(env, learner)
    if episode % 10 == 0:
        print(f'Episode: episode: {episode}')

    if episode % Hyperparameters.SAVE_MODEL == 0:
        path = f'{Hyperparameters.model_base_path}{Hyperparameters.model_base_name}{episode}'
        print("Model saved")
        torch.save(learner.policy_net.state_dict(), path)


    if len(learner.memory.memory) < Hyperparameters.MIN_MEMORY_SIZE:
        continue

    if episode % Hyperparameters.TRAIN_STEP == 0:
        learner.fit(Hyperparameters.EPOCHS)
        print(f'Average loss of last fit: {learner.avg_loss_per_fit[-1]}')

    if episode % Hyperparameters.TARGET_UPDATE == 0:
        learner.update_target_net()