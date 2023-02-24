

import random

def xover(offspring, cxpb, toolbox):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb: # cross two individuals with probability CXPB
            toolbox.mate(child1, child2) # fitness values of the children must be recalculated later
            del child1.fitness.values
            del child2.fitness.values


def mutation(offspring, mutpb, toolbox):
    for mutant in offspring:
        # mutate an individual with probability MUTPB
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values
