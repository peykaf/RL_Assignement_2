from gridworld import GridWorld
import random

class Passenger:
    
    def __init__(self):
        self.source = GridWorld.locations[random.choice(list(GridWorld.locations))]
        self.destination = GridWorld.locations[random.choice(list(GridWorld.locations))]
        print('Passenger...')
        print('  starting point: {}, destination point: {}'.format(self.source, self.destination))


    