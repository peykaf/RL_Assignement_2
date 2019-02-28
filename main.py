from gridworld import GridWorld
from taxi import Taxi
from passenger import Passenger

def main():
    gw = GridWorld()
    ps = Passenger()
    tx = Taxi()

    gw.display_grid(ps, tx)

if __name__ == "__main__":
    main()
