# Connect4 agent using Reinforcement Learning #

In this repository I have created an intelligent agent using reinforcement learning that learned how to play the game Connect4. I have experimented with DQN, minimax-DQN and DDQN algorithms.

The [ConnectX](https://www.kaggle.com/c/connectx/overview "Named link title") competition that is happening on Kaggle has motivated me to learn how to create this kind of agents. The most trained agents based on the code in this repository managed to achive top 15% ranking in the competition.

## Algorithm description ##

In the following part I will briefly explain the learning proccess of the agent and the code sections that make the alghorithm.

  * __Environment__ - this object represents dynamics of the Connect4 game. Each game of the Connect4 represents one episode in the Connect4 environment. The state of the game is represented as a matrix of size 6x7 where each element of the matrix has one of the following three values: _0_ - empty position in the board, _1_ - mark put by the first player and _2_ mark put by the second player. The environment objects has two main functions:
      * _reset()_ - resets the environment meaning that new game will be played.
      * _step(action)_ - executes the given action in the environment. After each executed action the agent receives the new state and a reward for the executed action.
  * __Model__ - represents the neural network that decides which actions the agent should execute. The goal of the whole project is to train this network in order for the agent to win more often in the Connect4 game. This network represents the Q function of the DQN algorithm.
  * __Experience__ - this object stores all the games that the agent has played which means that it represents the agent's experience of the game. Based on this experience the agent learns how to get better in the game.
  * __Exploration strategy__ - represents an implementation of a specific algorithm that tackles the trade-off between exploration and exploation in RL problems. I implemented the simples algorithm which is the _Epsilon Greedy Strategy_.
  * __Connect4 Learner__ - this object encapsulates the Model, Experience and the Exploration Strategy. Additionaly it implements the minimax-DQN and minimax-DDQN which is the main algorithm for training the _policy_network_. This object has one main function _fit(EPOCHS)_ which trains the policy network based on the stored experience for the given number of epochs.
  * __Self-play__ - this represents a function where an agent plays an episode in the environment against itself. The experience that is gathered during the episode is stored in the _Experience_ object.
  * __Hyperparameters__ - constants that are representing all the hyperparameters.
  * __Agent Evaluation__ - evaluates the agent performance against a random agent and a negamax agent.
  * __Agent submission__ - creates a submission file for the Kaggle's competition. Additionaly it adds two-step lookahead in order to provide better performance for the agent. The function encodes the weights of the policy network in a string using base64 encoding. 
