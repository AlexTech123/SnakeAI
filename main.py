from game.snake_game import SnakeGame
from agent.agent import Agent
import matplotlib.pyplot as plt
import pandas as pd


def train_model():
    scores = []
    game = SnakeGame()
    agent = Agent()
    n_games = 0
    while True:
        old_state = game.get_state()
        action = agent.get_action(old_state)
        reward, done, score = game.play_action(action)
        new_state = game.get_state()

        agent.train_short_memory(old_state, action, reward, new_state, done)

        if done:
            scores.append(score)
            n_games += 1
            game.reset_game()
            if n_games % 100 == 0:
                agent.epsilon /= 2
                print(n_games)
                df = pd.Series(scores)
                smoothed = df.rolling(window=20).mean()
                plt.plot(smoothed)
                plt.show()


if __name__ == "__main__":
    train_model()