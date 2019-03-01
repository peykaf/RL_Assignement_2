import numpy as np


class GridWorld:
    """
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: dropoff passenger

     actions:
    - 0: south
    - 1: north
    - 2: east
    - 3: west
    - 4: pickup
    - 5: dropoff

    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    
    def __init__(self, tx, ps):
        num_states = 500
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        self.num_actions = 6
        P = {state: {action: [] for action in range(self.num_actions)} for state in range(num_states)}
        self.sarsa(tx, ps)

    def sarsa(self, tx, ps):

        # initialize state
        state = (tx.start[0], tx.start[1], ps.source, ps.destination)
        encoded_state = self.encode(tx.start[0], tx.start[1], ps.source, ps.destination)

        # actions
        for actions in range(self.num_actions):
            pass


    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        return i