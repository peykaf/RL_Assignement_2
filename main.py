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
    solver = ['SARSA', 'Q_learning', 'expected_SARSA']
    temperature = [2, 3, 7]
    learning_rate = [0.3, 0.6, 0.9]

    gw = GridWorld(solver[1], temperature[1], learning_rate[2])


if __name__ == "__main__":
    main()
