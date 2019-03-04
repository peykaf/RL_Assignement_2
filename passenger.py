import random

class Passenger:

    locations = {'R': (0, 0),
                 'G': (0, 4),
                 'Y': (4, 0),
                 'B': (4, 3)}
    
    def __init__(self):
        # self.source = self.locations[random.choice(list(self.locations))]
        # self.destination = self.locations[random.choice(list(self.locations))]
        self.source = random.randint(0, 3)
        self.destination = random.randint(0, 3)
        self.locs = [(0,0), (0,4), (4,0), (4,3)]
        self.is_in_taxi = False
        print('Passenger...')
        print('  starting point: {} at {}'.format(self.source, self.locs[self.source]))
        print('  destination point: {} at {}'.format(self.destination, self.locs[self.destination]))


    