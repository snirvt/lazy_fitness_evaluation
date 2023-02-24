


'''The step_scheduler is a step function, where it calculates the ratio of fitness calls and then check what
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


'''The linear_scheduler is a piece-wise linear function, where it calculates the ratio of fitness calls and then 
check what fitness percentage is best suited.
The points parameter is a list of tuples where each tuple represent a ratio (first value) and a
corresponding fitness percentage (last value).
'''
def linear_scheduler(f_call, n_fitness_calls, points):
    points += [(1,1)]
    ratio = f_call/n_fitness_calls
    # Find the two points that ratio falls between
    left_point, right_point = None, None
    for i in range(len(points) - 1):
        if points[i][0] <= ratio <= points[i + 1][0]:
            left_point, right_point = points[i], points[i + 1]
            break
    # Calculate the slope and y-intercept of the line between the two points
    slope = (right_point[1] - left_point[1]) / (right_point[0] - left_point[0])
    y_intercept = left_point[1] - slope * left_point[0]
    # Calculate the interpolated value of x
    interpolated_value = slope * ratio + y_intercept
    return interpolated_value


if __name__ == "__main__":
    points = [(0,0.5),(0.2,0.6),(0.4, 0.75),(1,1)]
    print(linear_scheduler(0, 100, points))
    print(linear_scheduler(1, 100, points))
    print(linear_scheduler(20, 100, points))
    print(linear_scheduler(45, 100, points))
