import random


class Taxi:
    def __init__(self):
        self.location = (random.randint(0, 4), random.randint(0, 4))
        print('Taxi...')
        print('  located at: {}'. format(self.location))
