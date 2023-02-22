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
cxpb = 0.1  # Probability of performing crossover on an individual
mutpb = 0.75  # Probability of mutating an individual
ngen = 100  # Number of generations to evolve the population
n_fitness_calls = 100000


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
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1/INDIVIDUAL_SIZE)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalTSP_lazy, distances=distances, p=1)


pop = toolbox.population(n=POPULATION_SIZE)

p = 1
f_call = 0
hof = tools.HallOfFame(1)

fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    f_call += p
    ind.fitness.values = fit
hof.update(pop)
print("  Evaluated %i individuals" % len(pop))

# Extracting all the fitnesses of
fits = [ind.fitness.values[0] for ind in pop]
# Variable keeping track of the number of generations
g = 0
# Begin the evolutions
log = []
# while g < ngen:
while f_call < n_fitness_calls:
    # A new generation
    g = g + 1
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    hof.update(offspring)

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        # cross two individuals with probability CXPB
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            # fitness values of the children must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        # mutate an individual with probability MUTPB
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        f_call += p
        if f_call > n_fitness_calls:
            break
        ind.fitness.values = fit
    print("  Evaluated %i individuals" % len(invalid_ind))
    # The population is entirely replaced by the offspring
    pop[:] = offspring
    # Gather all the fitnesses in one list and print the stats
    valid_ind = [ind for ind in offspring if ind.fitness.valid]
    fits = [ind.fitness.values[0] for ind in valid_ind]
    gen_stats = {'mean': np.mean(fits), 'std': np.std(fits), 'min': min(fits), 'max': max(fits)}
    log.append(gen_stats)

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))



