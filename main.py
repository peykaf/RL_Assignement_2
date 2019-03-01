from gridworld import GridWorld
from taxi import Taxi
from passenger import Passenger
import random

def main():
    tx = Taxi()
    ps = Passenger()
    gw = GridWorld(tx, ps)


if __name__ == "__main__":
    main()
