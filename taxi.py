import random

class Taxi():

    actions = {'pickup': 1,
               'put_down': 0}

    
    def __init__(self):
        self.start = (random.randint(0, 4), random.randint(0, 4))
        self.actions = {'north': (-1, 0),
                        'south': (1, 0),
                        'east': (0, 1),
                        'west': (0, -1),
                        'pickup': (0, 0),
                        'putdown': (0, 0)}
        print('Taxi...')
        print('  located at: {}'. format(self.start))


