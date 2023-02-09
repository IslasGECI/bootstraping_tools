import numpy as np
import pandas as pd
from bootstrapping_tools import (
    boostrapping_feature,
    bootstrap_from_time_series,
    calculate_intervals_from_p_values_and_alpha,
    calculate_limits_from_p_values_and_alpha,
    calculate_p_values,
    choose_type_of_limits_from_p_values,
    generate_latex_interval_string,
    get_bootstrap_deltas,
    lambda_calculator,
    lambdas_bootstrap_from_dataframe,
    lambdas_from_bootstrap_table,
    mean_bootstrapped,
    power_law,
    remove_distribution_outliers,
    remove_outlier,
    seasons_from_date,
)


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


def test_lambdas_bootstrap_from_dataframe_removing_outliers():
    obtained_lambdas_bootstrap = lambdas_bootstrap_from_dataframe(
        data_nest, "Nest", N=20, remove_outliers=True
    )
    expected_lambdas_bootstrap = np.array([[1.795534, 1.821272, 1.848668]])
    are_close = np.isclose(expected_lambdas_bootstrap, obtained_lambdas_bootstrap, rtol=1e-5).all()
    assert are_close


def test_get_bootstrap_deltas():
    output = get_bootstrap_deltas(data_original)
    assert output == [0, 1, 0]


def test_bootstrap_from_time_series():
    obtained_bootstrap_from_time_series = bootstrap_from_time_series(
        data_nest, "Nest", N=100, remove_outliers=False
    )
    expected_bootstrap_from_time_series = np.array([1.77056253, 1.79716642, 1.82920025])
    are_close = np.isclose(
        expected_bootstrap_from_time_series, obtained_bootstrap_from_time_series, rtol=1e-5
    ).all()
    assert are_close, "Intervalo del 95% difiere"

    obtained_bootstrap_from_time_series = bootstrap_from_time_series(
        data_nest, "Nest", N=100, remove_outliers=False, alpha=0.1
    )
    expected_bootstrap_from_time_series = np.array([1.77824223, 1.79716642, 1.82171091])
    are_close = np.isclose(
        expected_bootstrap_from_time_series, obtained_bootstrap_from_time_series, rtol=1e-5
    ).all()
    assert are_close, "Intervalo del 90% difiere"


def test_calculate_limits_from_p_values_and_alpha():
    alpha = 0.1
    p_values = (0.727, 1 - 0.727)
    obtained_limits = calculate_limits_from_p_values_and_alpha(p_values, alpha)
    expected_limits = [5, 50, 95]
    assert obtained_limits == expected_limits
    p_values = (0.727, 0.05)
    obtained_limits = calculate_limits_from_p_values_and_alpha(p_values, alpha)
    expected_limits = [1, 50, 90]
    assert obtained_limits == expected_limits
    p_values = (0.05, 0.727)
    obtained_limits = calculate_limits_from_p_values_and_alpha(p_values, alpha)
    expected_limits = [10, 50, 99]
    assert obtained_limits == expected_limits


def test_choose_type_of_limits_from_p_values_and_alpha():
    p_values = (0.05, 0.727)
    alpha = 0.1
    type_of_limits = choose_type_of_limits_from_p_values(p_values, alpha)
    assert type_of_limits == "upper"
    p_values = (0.727, 0.05)
    type_of_limits = choose_type_of_limits_from_p_values(p_values, alpha)
    assert type_of_limits == "lower"
    p_values = (0.727, 1 - 0.727)
    type_of_limits = choose_type_of_limits_from_p_values(p_values, alpha)
    assert type_of_limits == "central"


def test_calculate_intervals_from_p_values_and_alpha():
    p_values = (0.727, 1 - 0.727)
    alpha = 0.1
    obtained_intervals = calculate_intervals_from_p_values_and_alpha(data_original, p_values, alpha)
    expected_intervals = np.array([1.0, 1.0, 1.65])
    are_close = np.isclose(obtained_intervals, expected_intervals, rtol=1e-5).all()
    assert are_close, "Intervalo del 90% difiere with one tail"


def test_calculate_p_values():
    expected_p_value = (0.0, 0.0625)
    output = calculate_p_values(data_original)
    assert expected_p_value == output


def test_generate_latex_interval_string():
    interval_from_original_data = np.array([1.77824223, 1.79716642, 1.82171091])

    expected_latex_deltas_string = "${2.0}_{-0.0}^{+0.0}$"
    obtained_latex_deltas_string = generate_latex_interval_string(interval_from_original_data)
    assert expected_latex_deltas_string == obtained_latex_deltas_string

    expected_latex_interval_string = "2.0 (2.0 - 2.0)"
    obtained_latex_interval_string = generate_latex_interval_string(
        interval_from_original_data, deltas=False
    )
    assert expected_latex_interval_string == obtained_latex_interval_string

    expected_latex_interval_string = "1.8 (1.78 - 1.82)"
    obtained_latex_interval_string = generate_latex_interval_string(
        interval_from_original_data, deltas=False, decimals=2
    )
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


def test_remove_outlier():
    data_original = np.array([1, 1, 2, 3, 7])
    expected_data = np.array([1, 1, 2, 3])
    obtained_data = remove_outlier("tukey", data_original)
    np.testing.assert_array_equal(expected_data, obtained_data)
    data_original = np.append(np.ones(45), [2, 5, 6])
    expected_data: np.array = np.append(np.ones(45), [2, 5])
    obtained_data = remove_outlier("std", data_original, number_of_std=5)
    np.testing.assert_array_equal(expected_data, obtained_data)


def test_remove_distribution_outliers():
    data_with_outliers = np.append(np.ones(100), np.arange(10))
    length_expected = 105
    obtained_data = remove_distribution_outliers(data_with_outliers)
    length_obtained_data = len(obtained_data)
    assert length_obtained_data == length_expected
