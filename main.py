from gridworld import GridWorld
from taxi import Taxi
from passenger import Passenger


def main():
    """
     solver:
    - 0 : SARSA
    - 1 : Q_learning
    - 2 : expected_SARSA
    """
    runs = 1
    segments = 1
    episodes = 1
    solver = ['SARSA', 'Q_learning', 'expected_SARSA']
    temperature = [1, 10, 100]
    learning_rate = [0.3, 0.6, 0.9]

    for run in range(runs):
        for segment in range(segments):
            for episode in range(episodes + 1):
                if episode < 9:  # training episodes
                    tx = Taxi()
                    ps = Passenger()
                    gw = GridWorld(tx, ps, solver[2], temperature[1], learning_rate[2])
                else:  # the 11th episode: you pick actions greedily based on the current value estimates
                    pass

if __name__ == "__main__":
    main()
