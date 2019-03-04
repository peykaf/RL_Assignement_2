import numpy as np
import copy


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

        # input
        num_states = 500
        num_rows = 5
        num_columns = 5
        self.max_row = num_rows - 1
        self.max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        self.num_actions = 6
        self.P = {state: {action: [] for action in range(self.num_actions)} for state in range(num_states)}
        self.Q = {state: {action: 0 for action in range(self.num_actions)} for state in range(num_states)}
        self.sarsa(tx, ps)

    def sarsa(self, tx, ps):
        # initialize state
        row = tx.location[0]
        col = tx.location[1]
        state = (row, col, ps.source, ps.destination)
        encoded_state = self.encode(row, col, ps.source, ps.destination)
        delivered = False

        # actions
        action = self.boltzmann_softmax()
        
        for action in range(self.num_actions):
            new_row, new_col, new_source = row, col, ps.source
            # south
            if action == 0:
                new_row = min(row + 1, self.max_row)
            elif action == 1:
                new_row = max(row - 1, 0)
            elif action == 2:
                new_col = min(col + 1, self.max_col)
            elif action == 3:
                new_col = max(col - 1, 0)
            elif action == 4:  # pickup
                if tx.location == ps.source:
                    ps.is_in_taxi = True
                else:  # illegal pickup
                    reward = -10
            elif action == 5:
                if tx.location == ps.destination:
                    reward = 20
                    delivered = True
                else:  # illegal putdown
                    reward = -10
        new_state = (new_row, new_col, ps.source, ps.destination)
        new_encoded_state = self.encode(new_row, new_col, ps.source, ps.destination)
        self.P[state][action].append(new_state, reward, delivered)


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

    def boltzmann_softmax(self):
        # choose action by boltzmann
        # above is deterministic
        pass