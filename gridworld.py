import numpy as np

class GridWorld:
    
    locations = {'R': (0, 0),
                  'G': (0, 4),
                  'Y': (4, 0),
                  'B': (4, 3)}

    def __init__(self):
        self.size = (5, 5)
        self.grid = self._make_grid()
        self.rewards = self._reward_grid()

    def display_grid(self, Passenger, Taxi):
        print('+---------+')
        print('|R: | : :G|')
        print('| : : : : |')
        print('| : : : : |')
        print('| | : | : |')
        print('|Y| : |B: |')
        print('+---------+')

        for i, j in np.ndenumerate(np.array((5, 5))):
            print()



    def _make_grid(self):
        pass

    def _reward_grid(self):
        rewards = {'per_action': -1,
                   'successful_delivery': 20,
                   'illegal_attempt': -10}

        return rewards

    def allowed_movements(self, current, next):
        if current == (0, 1) and next == (0, 2) or current == (0, 2) and next == (0, 1) or \
            current == (1, 1) and next == (1, 2) or current == (1, 2) and next == (1, 1) or \
            current == (3, 0) and next == (3, 1) or current == (3, 1) and next == (3, 0) or \
            current == (4, 0) and next == (4, 1) or current == (4, 1) and next == (4, 0) or \
            current == (3, 2) and next == (3, 3) or current == (3, 3) and next == (3, 2) or \
            current == (4, 2) and next == (4, 3) or current == (4, 3) and next == (4, 2):

            next = current

        return next