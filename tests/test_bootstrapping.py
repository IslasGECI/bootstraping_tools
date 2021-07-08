import numpy as np
from bootstrapping_tools import (
    seasons_from_date,
    power_law,
    remove_distribution_outliers,
    boostrapping_feature,
    get_bootstrap_interval,
    calculate_p_values,
    lambda_calculator,
    lambdas_bootstrap_from_dataframe,
    lambdas_from_bootstrap_table,
    generate_latex_interval_string,
    mean_bootstrapped,
    tukey_fences,
    bootstrap_from_time_series,
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


def test_tukey_fences():
    data_original = np.append(np.ones(2), [2, 3, 6])
    expected_data: np.array = np.append(np.ones(2), [2, 3])
    obtained_data = tukey_fences(data_original)
    np.testing.assert_array_equal(expected_data, obtained_data)


def test_boostrapping_feature():
    output = boostrapping_feature(data_original, number_sample=2)
    assert output == [1.0, 1.0]


def test_lambdas_from_bootstrap_table():
    data_nest = pd.DataFrame({"2018": [1.1, 1.1, 0.9, 0.9], "2019": [2.1, 2.1, 1.9, 1.9]})
    obtained_lambdas = lambdas_from_bootstrap_table(data_nest)
    expected_lambdas = [1.9091, 1.9091, 2.1111, 2.1111]
    are_close = np.isclose(expected_lambdas, obtained_lambdas, rtol=1e-5).all()
    assert are_close


data_nest = pd.DataFrame(
    {
        "Temporada": [2018, 2019, 2020, 2018, 2019, 2020, 2018, 2019, 2020],
        "Nest": [2.0, 3.9, 6.9, 2.1, 4.0, 7.0, 1.9, 3.8, 6.8],
    }
)


def test_lambdas_bootstrap_from_dataframe():
    obtained_lambdas_bootstrap = lambdas_bootstrap_from_dataframe(
        data_nest, "Nest", N=20, remove_outliers=False
    )
    expected_lambdas_bootstrap = np.array([[1.795534, 1.821272, 1.848668]])
    are_close = np.isclose(expected_lambdas_bootstrap, obtained_lambdas_bootstrap, rtol=1e-5).all()
    assert are_close


def test_get_bootstrap_interval():
    output = get_bootstrap_interval(data_original)
    assert output == [0, 1, 0]


def test_bootstrap_from_time_series():
    obtained_bootstrap_from_time_series = bootstrap_from_time_series(
        data_nest, "Nest", N=20, remove_outliers=False
    )
    expected_bootstrap_from_time_series = np.array([1.78180307, 1.82117423, 1.94509028])
    are_close = np.isclose(
        expected_bootstrap_from_time_series, obtained_bootstrap_from_time_series, rtol=1e-5
    ).all()
    assert are_close


def test_calculate_p_values():
    expected_p_value = (0.0, 0.0625)
    output = calculate_p_values(data_original)
    assert expected_p_value == output


def test_generate_latex_interval_string():
    expected_latex_interval_string = "${1.0}_{-0.0}^{+0.0}$"
    obtained_latex_interval_string = generate_latex_interval_string(data_original)
    assert expected_latex_interval_string == obtained_latex_interval_string


def test_mean_bootstrapped():
    data_test = np.arange(0, 10)
    N_test = 5
    obtained_distribution = mean_bootstrapped(data_test, N=N_test)
    expected_distribution = [4.1, 5.0, 5.1, 6.2, 6.1]
    np.testing.assert_array_equal(obtained_distribution, expected_distribution)
    obtained_distribution = mean_bootstrapped(data_test)
    default_bootstrapping_size_sample = 2000
    assert len(obtained_distribution) == default_bootstrapping_size_sample
