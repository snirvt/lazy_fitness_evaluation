
import sys
print(sys.prefix)
import os.path

import random
import numpy as np
from deap import algorithms, base, creator, tools

import array
import random

from evolution_helper import xover, mutation
from utils import generate_distances
from fitness_handler import evalTSP, evalTSP_lazy
from schedulers import step_scheduler, linear_scheduler
import argparse


import warnings
warnings.filterwarnings("ignore")


argparser = argparse.ArgumentParser()
argparser.add_argument('--INDIVIDUAL_SIZE', type=int, default=20, help='')#, required=False)
argparser.add_argument('--POPULATION_SIZE', type=int, default=100, help='')
argparser.add_argument('--cxpb', type=float, default=0.1, help='cross over percentage')
argparser.add_argument('--mutpb', type=float, default=1.0, help='mutation percentage')
argparser.add_argument('--n_fitness_calls', type=int, default=100000, help='max fitness calls')
argparser.add_argument('--novelty_param', type=int, default=1, help='')
argparser.add_argument('--scheduler', choices=['step_scheduler', 'linear_scheduler'], default=step_scheduler, help='')
argparser.add_argument('--points', type=list, default=[(0, 0.8),(0.2, 1)], help='scheduler function argument')

args, unknown = argparser.parse_known_args()








# Define problem parameters
INDIVIDUAL_SIZE = args.INDIVIDUAL_SIZE  # Number of genes in each individual
NUM_CITIES = INDIVIDUAL_SIZE  # Number of cities in the TSP problem
POPULATION_SIZE = args.POPULATION_SIZE  # Number of individuals in the population
cxpb = args.cxpb  # Probability of performing crossover on an individual
mutpb = args.mutpb  # Probability of mutating an individual
n_fitness_calls = args.n_fitness_calls
novelty_param = args.novelty_param

scheduler = step_scheduler
if args.scheduler == 'linear_scheduler':
    scheduler = linear_scheduler

points = args.points



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

# p = 1
f_call = 0
p = scheduler(f_call, n_fitness_calls, points)
hof = tools.HallOfFame(novelty_param)
toolbox.register("evaluate", evalTSP_lazy, distances=distances, p=p)
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    f_call += p
    ind.fitness.values = fit
hof.update(pop)
fits = [ind.fitness.values[0] for ind in pop]
g = 0
log = []
while f_call < n_fitness_calls:
    g = g + 1
    print(f"-- Generation {g}, p {p}--")
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    hof.update(offspring)
    xover(offspring, cxpb, toolbox)
    mutation(offspring, mutpb, toolbox)
    if novelty_param > 0:
        pop[:novelty_param] = hof.items[:]
    p = scheduler(f_call, n_fitness_calls, points)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    toolbox.register("evaluate", evalTSP_lazy, distances=distances, p=p)
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        f_call += p
        if f_call > n_fitness_calls:
            break
        ind.fitness.values = fit
    pop[:] = offspring
    if novelty_param > 0:
        pop[:novelty_param] = hof.items[:]
    valid_ind = [ind for ind in offspring if ind.fitness.valid]
    fits = [ind.fitness.values[0] for ind in valid_ind]
    gen_stats = {'mean': np.mean(fits), 'std': np.std(fits), 'min': min(fits), 'max': max(fits)}
    log.append(gen_stats)


best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
toolbox.register("evaluate", evalTSP, distances=distances)
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

