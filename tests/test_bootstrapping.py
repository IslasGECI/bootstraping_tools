import numpy as np
from bootstrapping_tools import (
    seasons_from_date,
    power_law,
    remove_distribution_outliers,
    boostrapping_feature,
    get_bootstrap_interval,
    calculate_p_values,
    lambda_calculator,
    lambdas_from_bootstrap_table,
    generate_latex_interval_string,
    lambdas_bootstrap_from_dataframe,
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
    obtained_data = remove_distribution_outliers(data_original, number_of_std=5)
    np.testing.assert_array_equal(expected_data, obtained_data)


def test_boostrapping_feature():
    output = boostrapping_feature(data_original, N=2)
    assert output == [1.0, 1.0]


def test_lambdas_from_bootstrap_table():
    data_nest = pd.DataFrame({"2018": [1.1, 1.1, 0.9, 0.9], "2019": [2.1, 2.1, 1.9, 1.9]})
    obtained_lambdas = lambdas_from_bootstrap_table(data_nest)
    expected_lambdas = [1.9091, 1.9091, 2.1111, 2.1111]
    are_close = np.isclose(expected_lambdas, obtained_lambdas, rtol=1e-5).all()
    assert are_close


def test_lambdas_bootstrap_from_dataframe():
    data_nest = pd.DataFrame(
        {
            "Temporada": [2018, 2019, 2020, 2018, 2019, 2020, 2018, 2019, 2020],
            "Nest": [2.0, 3.9, 6.9, 2.1, 4.0, 7.0, 1.9, 3.8, 6.8],
        }
    )
    lambdas_bootstrap_from_dataframe(data_nest, "Nest", N=20, remove_outliers=False)


def test_get_bootstrap_interval():
    output = get_bootstrap_interval(data_original)
    assert output == [0, 1, 0]


def test_calculate_p_values():
    expected_p_value = (0.0, 0.0625)
    output = calculate_p_values(data_original)
    assert expected_p_value == output


def test_generate_latex_interval_string():
    expected_latex_interval_string = "${1.0}_{-0.0}^{+0.0}$"
    obtained_latex_interval_string = generate_latex_interval_string(data_original)
    assert expected_latex_interval_string == obtained_latex_interval_string
