


'''The step_scheduler is a step function, where it calculates the ratio of calls and then check what
fitness percentage is best suited.
The steps parameter is a list of tuples where each tuple represent a cumulative proportion (first value) and a
corresponding fitness percentage (last value).
The default is p=1
When the ratio of fitness calls surpassed the cumulative proportions it will return 1.
'''
def step_scheduler(f_call, n_fitness_calls, steps = [(0.2,0.5),(0.2,0.75)]):
    ratio = f_call/n_fitness_calls
    sum_p = 0
    for step in steps:
        sum_p += step[0]
        if ratio <= sum_p:
            return step[1]
    return 1




