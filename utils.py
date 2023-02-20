

import numpy as np
import random
def generate_distances(NUM_CITIES):
    distances = np.zeros((NUM_CITIES, NUM_CITIES))
    for city in range(NUM_CITIES):
        for to_city in range(city+1, NUM_CITIES):
            distances[to_city][city] = \
            distances[city][to_city] = random.random()
    return distances