from model.model import Net
from model.trainer import QTrainer
import numpy as np
import torch
from collections import deque

class Agent:

    def __init__(self):
        self.lr = 0.001
        self.epsilon = 0.1  # randomness
        self.gamma = 0.9  # discount rate
        self.model = Net(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0, 0, 0]
        if np.random.random() < self.epsilon:
            move = np.random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0.unsqueeze(0))
            move = torch.argmax(prediction.squeeze(0)).item()
            final_move[move] = 1

        return final_move