import random
from torch.utils.data import DataLoader
from utils import create_dual_memory

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, state, action, next_state, reward):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_sample(self, batch_size):
        return batch_size < len(self.memory)

    def dataloader(self, batch_size, percent=0.5):
        ms = int(len(self.memory) * percent)
        sample_memory = self.sample(ms)
        dataloader = DataLoader(sample_memory, batch_size, shuffle=True, drop_last=False)
        sample_memory = create_dual_memory(sample_memory)
        dataloader_dual = DataLoader(sample_memory, batch_size, shuffle=True, drop_last=False)
        return (dataloader, dataloader_dual)

    def __len__(self):
        return len(self.memory)