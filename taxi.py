from gridworld import GridWorld
import random

class Taxi():

    navigation_actions = {'north': (-1, 0),
                          'south': (1, 0),
                          'east': (0, 1),
                          'west': (0, -1)}
    
    actions = {'pickup': 1,
               'put_down': 0}

    
    def __init__(self):
        self.start = (random.randint(0, 4), random.randint(0, 4))
        print('Taxi...')
        print('  located at: {}'. format(self.start))
