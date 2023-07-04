from bootstrapping_tools import get_percentile


def tests_get_percentile():
    lambdas_n0s = [(3.1, 5), (1.2, 3), (2, 4)]
    limits = [2.5, 50, 97.5]
    obtained = get_percentile(lambdas_n0s, limits)
    expected = [(1.2, 3), (2, 4), (3.1, 5)]
    assert obtained == expected

    lambdas_n0s = [(3.1, 5), (4, 6), (1.2, 3), (2, 4), (0, 7)]
    obtained = get_percentile(lambdas_n0s, limits)
    expected = [(0, 7), (2, 4), (4, 6)]
    assert obtained == expected

    lambdas_n0s = [3.1, 4, 1.2, 2, 0]
    obtained = get_percentile(lambdas_n0s, limits)
    expected = [0, 2, 4]
    assert obtained == expected

    lambdas_n0s = [3.1, 4, 1.2, 0]
    obtained = get_percentile(lambdas_n0s, limits)
    expected = [0, 3.1, 4]
    assert obtained == expected

    lambdas_n0s = [3.1, 4, 1.2, 2, 0, 5.5]
    obtained = get_percentile(lambdas_n0s, limits)
    expected = [0, 2, 5.5]
    assert obtained == expected

    lambdas_n0s = [(0, 7), (0, 7), (0, 3), (3.1, 5), (4, 6), (1.2, 3), (2, 4), (0, 7)]
    obtained = get_percentile(lambdas_n0s, limits)
    expected = [(0, 3), (1.2, 3), (4, 6)]
    assert obtained == expected
