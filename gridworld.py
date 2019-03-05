import numpy as np
import random


class GridWorld:
    """
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: putdown passenger

     actions:
    - 0: south
    - 1: north
    - 2: east
    - 3: west
    - 4: pickup
    - 5: putdown

     locations: (for pickup and putdown of passenger)
    - 0: 'R': (0, 0)
    - 1: 'G': (0, 4)
    - 2: 'Y': (4, 0)
    - 3: 'B': (4, 3)
    - 4:  passenger is in taxi

    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    def __init__(self, tx, ps, solver, temperature, learning_rate):
        # input
        grid_map = [
            "+---------+",
            "|R: | : :G|",
            "| : : : : |",
            "| : : : : |",
            "| | : | : |",
            "|Y| : |B: |",
            "+---------+",
        ]

        self.grid = np.asarray(grid_map, dtype='c')

        self.temperature = temperature
        self.learning_rate = learning_rate

        num_states = 500
        num_rows = 5
        num_columns = 5
        self.max_row = num_rows - 1
        self.max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        self.num_actions = 6
        self.P = {state: {action: [] for action in range(self.num_actions)} for state in range(num_states)}
        Q = {state: {action: random.randint(0, 0) for action in range(self.num_actions)} for state in range(num_states)}
        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

        print('====================================================')
        print('Solving with...{}'.format(solver))
        print('====================================================')

        if solver == 'SARSA':
            self.sarsa(tx, ps, Q)
        elif solver == 'Q_learning':
            self.q_learning(tx, ps, Q)
        else:  # expected_SARSA
            self.expected_sarsa(tx, ps, Q)


    def sarsa(self, tx, ps, Q):
        # initialize state
        row = tx.location[0]
        col = tx.location[1]
        state = (row, col, ps.source, ps.destination)
        encoded_state = self.encode(row, col, ps.source, ps.destination)

        # actions
        action, _ = self.boltzmann_softmax(encoded_state, Q, temperature=self.temperature)

        while ps.delivered != True:
            # *************
            print('S = ({}, {}) - {}'.format(row, col, encoded_state))
            print('a = {}'.format(action))

            new_row, new_col, reward = self.action_effects(action, row, col, tx, ps)
            new_state = (new_row, new_col, ps.location, ps.destination)
            new_encoded_state = self.encode(new_row, new_col, ps.source, ps.destination)
            # *************
            print("S' = ({}, {}) - {}".format(new_row, new_col, new_encoded_state))
            print('r = {}'.format(reward))

            new_action, _ = self.boltzmann_softmax(new_encoded_state, Q, temperature=self.temperature)
            # *************
            print("a' = {}".format(new_action))
            Q = self.update_Q_sarsa(encoded_state, action, new_encoded_state, new_action,
                              reward, Q, learning_rate=self.learning_rate, discount=0.9)

            if ps.is_in_taxi == True:
                ps.location = 4
                print('Passenger picked up at {}'.format(self.locs[ps.source]))
            print('----------------------------------------------------')
            row, col = new_row, new_col
            encoded_state = self.encode(row, col, ps.location, ps.destination)
            tx.location = (new_row, new_col)
            action = new_action

        print('====================================================')
        print('Episode Completed...')
        print('passenger picked up at location: {}'.format(self.locs[ps.source]))
        print('Passenger dropped at the destination: {}'.format(self.locs[ps.destination]))
        print('====================================================')

    def q_learning(self, tx, ps, Q):
        # initialize state
        row = tx.location[0]
        col = tx.location[1]
        state = (row, col, ps.source, ps.destination)
        encoded_state = self.encode(row, col, ps.source, ps.destination)

        # loop for each step of episode
        while ps.delivered != True:
            action, _ = self.boltzmann_softmax(encoded_state, Q, temperature=self.temperature)  # select action a in state S
            # *************
            print('S = ({}, {}) - {}'.format(row, col, encoded_state))
            print('a = {}'.format(action))

            new_row, new_col, reward = self.action_effects(action, row, col, tx, ps)  # observe R and S'
            new_state = (new_row, new_col, ps.location, ps.destination)  # define S'
            new_encoded_state = self.encode(new_row, new_col, ps.source, ps.destination)  # encode S'
            # *************
            print("S' = ({}, {}) - {}".format(new_row, new_col, new_encoded_state))
            print('r = {}'.format(reward))

            Q = self.update_Q_q_learning(encoded_state, action, new_encoded_state,
                                    reward, Q, learning_rate=self.learning_rate, discount=0.9)
            if ps.is_in_taxi == True:
                ps.location = 4
                print('Passenger picked up at {}'.format(self.locs[ps.source]))
            print('----------------------------------------------------')
            row, col = new_row, new_col
            encoded_state = self.encode(row, col, ps.location, ps.destination)
            tx.location = (new_row, new_col)

        print('====================================================')
        print('Episode Completed...')
        print('passenger picked up at location: {}'.format(self.locs[ps.source]))
        print('Passenger dropped at the destination: {}'.format(self.locs[ps.destination]))
        print('====================================================')

    def expected_sarsa(self, tx, ps, Q):
        # initialize state
        row = tx.location[0]
        col = tx.location[1]
        state = (row, col, ps.source, ps.destination)
        encoded_state = self.encode(row, col, ps.source, ps.destination)

        # loop for each step of episode
        while ps.delivered != True:
            action, prob = self.boltzmann_softmax(encoded_state, Q, temperature=self.temperature)  # select action a in state S
            # *************
            print('S = ({}, {}) - {}'.format(row, col, encoded_state))
            print('a = {}'.format(action))

            new_row, new_col, reward = self.action_effects(action, row, col, tx, ps)  # observe R and S'
            new_state = (new_row, new_col, ps.location, ps.destination)  # define S'
            new_encoded_state = self.encode(new_row, new_col, ps.source, ps.destination)  # encode S'
            # *************
            print("S' = ({}, {}) - {}".format(new_row, new_col, new_encoded_state))
            print('r = {}'.format(reward))

            Q = self.update_Q_expected_sarsa(encoded_state, action, new_encoded_state,
                                         reward, Q, prob, learning_rate=self.learning_rate, discount=0.9)
            if ps.is_in_taxi == True:
                ps.location = 4
                print('Passenger picked up at {}'.format(self.locs[ps.source]))
            print('----------------------------------------------------')
            row, col = new_row, new_col
            encoded_state = self.encode(row, col, ps.location, ps.destination)
            tx.location = (new_row, new_col)

        print('====================================================')
        print('Episode Completed...')
        print('passenger picked up at location: {}'.format(self.locs[ps.source]))
        print('Passenger dropped at the destination: {}'.format(self.locs[ps.destination]))
        print('====================================================')

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

    def boltzmann_softmax(self, encoded_state, Q, temperature):
        prob_t = [0] * 6
        for a in range(6):
            prob_t[a] = np.exp(Q[encoded_state][a] / temperature)
        # numpy matrix element-wise division for denominator (sum of numerators)
        prob_t = np.true_divide(prob_t, sum(prob_t))
        action = np.random.choice(np.arange(6), p=prob_t)

        return action, prob_t

    def action_effects(self, action, row, col, tx, ps):
        new_row, new_col = row, col

        if action == 0:  # south
            new_row = min(row + 1, self.max_row)
            reward = -1
        elif action == 1:  # north
            new_row = max(row - 1, 0)
            reward = -1
        elif action == 2:  # east
            if self.grid[1 + row, 2 * col + 2] == b":":  # allowed movement
                new_col = min(col + 1, self.max_col)
                reward = -1
            else:
                reward = -1
        elif action == 3:  # west
            if self.grid[1 + row, 2 * col ] == b":":  # allowed movement
                new_col = max(col - 1, 0)
                reward = -1
            else:
                reward = -1
        elif action == 4:  # pickup
            if tx.location == self.locs[ps.source] and ps.location < 4:
                ps.is_in_taxi = True
                reward = -1
            else:  # illegal pickup
                reward = -10
        else:  # putdown (action = 5)
            if tx.location == self.locs[ps.destination] and ps.is_in_taxi == True:
                reward = 20
                ps.delivered = True
            else:  # illegal putdown
                reward = -10

        return new_row, new_col, reward

    def update_Q_sarsa(self, state, action, new_state, new_action, reward, Q, learning_rate, discount):
        target = reward + discount * Q[new_state][new_action]
        Q[state][action] += learning_rate * (target - Q[state][action])

        return Q

    def update_Q_q_learning(self, state, action, new_state, reward, Q, learning_rate, discount):
        target = reward + discount * (max(Q[new_state].values()))
        Q[state][action] += learning_rate * (target - Q[state][action])

        return Q

    def update_Q_expected_sarsa(self, state, action, new_state, reward, Q, prob, learning_rate, discount):
        target = reward + discount * (sum(prob[a] * Q[new_state][a] for a in range(6)))
        Q[state][action] += learning_rate * (target - Q[state][action])

        return Q