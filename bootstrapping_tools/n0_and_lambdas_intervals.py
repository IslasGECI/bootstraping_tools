def get_percentile(lambdas_n0s, limits):
    sorted_tuples = sorted(lambdas_n0s)
    divided_limits = [limit / 100 for limit in limits]
    length_list = len(lambdas_n0s) - 1
    indexes = [round(limit * length_list) for limit in divided_limits]
    return [sorted_tuples[i] for i in indexes]
