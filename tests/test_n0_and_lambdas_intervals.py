def tests_get_percentile():
    lambdas_n0s = [(1.2, 3), (2, 4), (3.1, 5)]
    limits = [2.5, 50, 97.5]
    get_percentile(lambdas_n0s, limits)
