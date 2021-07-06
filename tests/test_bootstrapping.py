import numpy as np
from bootstraping_tools import (
    seasons_from_date,
    power_law,
    remove_distribution_outliers,
    boostrapping_feature,
    get_bootstrap_interval,
    calculate_p_values,
    lambda_calculator,
    lambdas_from_bootstrap_table,
)
import pandas as pd


def test_seasons_from_date():
    input_date = pd.DataFrame(
        ["13/May/2019", "16/May/2018", "23/Abr/2020", "25/Abr/2018"], columns=["Fecha"]
    )
    expected = np.array(["2019", "2018", "2020", "2018"])
    obtained = seasons_from_date(input_date)
    np.testing.assert_array_equal(obtained, expected)


step_time: int = 1
Lambda = 2
No: int = 1


def test_power_law():
    output = power_law(step_time, Lambda, No)
    assert output == 2


data_original = np.append(np.ones(45), [2, 5, 6])


def test_lambda_calculator():
    seasons = [1, 2]
    nest = [1, 2]
    obtained_parameters = lambda_calculator(seasons, nest)
    expected_parameters = [2.0, 1.0]
    np.testing.assert_almost_equal(expected_parameters, obtained_parameters, decimal=4)


def test_remove_distribution_outliers():
    expected_data: np.array = np.append(np.ones(45), [2, 5])
    obtained_data = remove_distribution_outliers(data_original)
    np.testing.assert_array_equal(expected_data, obtained_data)


def test_boostrapping_feature():
    output = boostrapping_feature(data_original, N=2)
    assert output == [1.0, 1.0]


def test_lambdas_from_bootstrap_table():
    data_nest = pd.DataFrame({"2018": [1, 1], "2019": [2, 2]})
    lambdas_from_bootstrap_table(data_nest)


def test_get_bootstrap_interval():
    output = get_bootstrap_interval(data_original)
    assert output == [0, 1, 0]


def test_calculate_p_values():
    expected_p_value = (0.0, 0.0625)
    output = calculate_p_values(data_original)
    assert expected_p_value == output
