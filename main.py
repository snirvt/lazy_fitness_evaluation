import random
import numpy as np
from deap import algorithms, base, creator, tools

import array
import random

from utils import generate_distances
from fitness_handler import evalTSP, evalTSP_lazy

# Define problem parameters
NUM_CITIES = 20  # Number of cities in the TSP problem
INDIVIDUAL_SIZE = NUM_CITIES  # Number of genes in each individual
POPULATION_SIZE = 100  # Number of individuals in the population
cxpb = 0.8  # Probability of performing crossover on an individual
mutpb = 0.2  # Probability of mutating an individual
ngen = 100  # Number of generations to evolve the population

distances = generate_distances(NUM_CITIES)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("indices", random.sample, range(INDIVIDUAL_SIZE), INDIVIDUAL_SIZE)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalTSP_lazy, distances=distances)

pop = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                               stats=stats, halloffame=hof, verbose=True)

# TODO change p during evolution
