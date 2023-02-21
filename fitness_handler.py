
import random

def evalTSP(individual, distances):
    distance = distances[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distances[gene1][gene2]
    return distance,


def evalTSP_lazy(individual, distances, p=0.5):
    start_idx = random.sample(range(len(individual)), 1)[0]
    end_idx = start_idx + int(len(individual) * p)
    double_individual = list(individual) + list(individual)
    distance = 0
    for gene1, gene2 in zip(double_individual[start_idx:end_idx], double_individual[start_idx+1:end_idx+1]):
        distance += distances[gene1][gene2]
    return distance,