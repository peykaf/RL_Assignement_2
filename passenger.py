import random


class Passenger:
    """
     locations: (for pickup and putdown of passenger)
    - 0: 'R': (0, 0)
    - 1: 'G': (0, 4)
    - 2: 'Y': (4, 0)
    - 3: 'B': (4, 3)
    - 4:  passenger is in taxi
    """
    locations = {'R': (0, 0),
                 'G': (0, 4),
                 'Y': (4, 0),
                 'B': (4, 3)}
    
    def __init__(self):
        self.source = random.randint(0, 3)
        self.destination = random.randint(0, 3)
        # self.source = 2
        # self.destination = 1
        self.location = self.source
        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.is_in_taxi = False
        self.delivered = False
        # print('Passenger...')
        # print('  starting point: {} at {}'.format(self.source, self.locs[self.source]))
        # print('  destination point: {} at {}'.format(self.destination, self.locs[self.destination]))
