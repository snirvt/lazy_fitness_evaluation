

import numpy as np
import random
import os


def generate_distances(NUM_CITIES, seed=None):
    random.seed(seed)
    distances = np.zeros((NUM_CITIES, NUM_CITIES))
    for city in range(NUM_CITIES):
        for to_city in range(city+1, NUM_CITIES):
            distances[to_city][city] = \
            distances[city][to_city] = random.random()
    return distances


def make_dir(directory_name):
    if not directory_name:
        return ''
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    if directory_name[-1] != '/':
        directory_name = directory_name + '/'
    return directory_name

def empty_list():
    return []