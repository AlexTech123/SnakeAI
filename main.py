import torch

from game.snake_game import SnakeGame
from agent.agent import Agent
import matplotlib.pyplot as plt
import pandas as pd
import os


def train_model():
    scores = []
    game = SnakeGame()
    agent = Agent()
    n_games = 0
    record = 0
    while True:
        old_state = game.get_state()
        action = agent.get_action(old_state)
        reward, done, score = game.play_action(action)
        new_state = game.get_state()

        agent.train_short_memory(old_state, action, reward, new_state, done)

        if done:
            if score > record:
                print("New score:", score)
                record = score

            scores.append(score)
            n_games += 1
            game.reset_game()

            if n_games % 100 == 0:
                agent.epsilon /= 2
                df = pd.Series(scores)
                smoothed = df.rolling(window=20).mean()
                plt.plot(smoothed)
                plt.show()

            if n_games == 550:
                agent.model.save()

def play():
    game = SnakeGame()
    game.speed = 20

    agent = Agent()
    agent.epsilon = 0
    agent.model.load_state_dict(torch.load('./model/model.pth', weights_only=True))

    while True:
        old_state = game.get_state()
        action = agent.get_action(old_state)
        reward, done, score = game.play_action(action)

        if done:
            game.reset_game()


if __name__ == "__main__":
    if not os.path.exists('./model/model.pth'):
        train_model()
    else:
        play()