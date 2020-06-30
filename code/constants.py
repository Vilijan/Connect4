class Hyperparameters:
    BATCH_SIZE = 64
    EPOCHS = 4

    TRAIN_STEP = 50
    TARGET_UPDATE = 100

    EPS_START = 1
    EPS_END = 0.01
    EPS_DECAY = 0.001

    GAMMA = 0.999
    LEARNING_RATE = 0.001
    MIN_LEARNING_RATE = 0.000001

    SAVE_MODEL = 1000

    MEMORY_SIZE = 40000
    MIN_MEMORY_SIZE = 5000

    EPISODES = 10000

    model_base_path = "models/"
    model_base_name = "ddqn_cnn_"