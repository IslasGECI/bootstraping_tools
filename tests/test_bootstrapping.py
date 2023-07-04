import numpy as np
import pandas as pd
from bootstrapping_tools import (
    boostrapping_feature,
    bootstrap_from_time_series,
    xxbootstrap_from_time_series,
    calculate_bootstrapped_mean,
    calculate_intervals_from_p_values_and_alpha,
    calculate_limits_from_p_values_and_alpha,
    calculate_p_values,
    choose_type_of_limits_from_p_values,
    generate_latex_interval_string,
    get_bootstrap_deltas,
    lambda_calculator,
    lambda_calculator_from_resampled_data,
    lambdas_bootstrap_from_dataframe,
    lambdas_from_bootstrap_table,
    mean_bootstrapped,
    power_law,
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


def test_lambda_calculator_from_resampled_data():
    seasons = [2, 1, 0, 3, 2, 6, 5]
    nest = [4, 2, 1, 8, 4, 64, 32]
    obtained_parameters_with_first_season = lambda_calculator_from_resampled_data(seasons, nest)
    expected_parameters = [2.0, 1.0]
    np.testing.assert_almost_equal(
        expected_parameters, obtained_parameters_with_first_season, decimal=4
    )

    seasons = [2, 1, 3, 2, 6, 5]
    nest = [4, 2, 8, 4, 64, 32]
    obtained_parameters_without_first_season = lambda_calculator_from_resampled_data(seasons, nest)
    np.testing.assert_almost_equal(
        obtained_parameters_with_first_season, obtained_parameters_without_first_season, decimal=4
    )


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
        data_nest,
        "Nest",
        N=20,
    )
    expected_lambdas_bootstrap = np.array([[1.795534, 1.821272, 1.848668]])
    are_close = np.isclose(expected_lambdas_bootstrap, obtained_lambdas_bootstrap, rtol=1e-5).all()
    assert are_close
    _, obtained_lambdas_bootstrap = lambdas_bootstrap_from_dataframe(
        data_nest,
        "Nest",
        N=20,
        return_distribution=True,
    )
    are_close = np.isclose(expected_lambdas_bootstrap, obtained_lambdas_bootstrap, rtol=1e-6).all()
    assert are_close


def test_get_bootstrap_deltas():
    output = get_bootstrap_deltas(data_original)
    assert output == [0, 1, 0]


def test_bootstrap_from_time_series():
    data_nest = pd.DataFrame(
        {
            "Temporada": [2019, 2020, 2018, 2018, 2019, 2020, 2018, 2019, 2020],
            "Nest": [3.9, 6.9, 2.0, 2.1, 4.0, 7.0, 1.9, 3.8, 6.8],
        }
    )
    obtained_bootstrap_from_time_series = xxbootstrap_from_time_series(
        data_nest,
        "Nest",
        N=100,
    )
    print(obtained_bootstrap_from_time_series)
    expected_bootstrap_from_time_series = np.array([1.76982509, 1.80925651, 1.83916692])
    are_close = np.isclose(
        expected_bootstrap_from_time_series, obtained_bootstrap_from_time_series, rtol=1e-5
    ).all()
    assert are_close, "Intervalo del 95% difiere"

    obtained_bootstrap_from_time_series = bootstrap_from_time_series(
        data_nest, "Nest", N=100, alpha=0.1
    )
    print(obtained_bootstrap_from_time_series)
    expected_bootstrap_from_time_series = np.array([1.76982509, 1.80925651, 1.83715074])
    are_close = np.isclose(
        expected_bootstrap_from_time_series, obtained_bootstrap_from_time_series, rtol=1e-5
    ).all()
    assert are_close, "Intervalo del 90% difiere"

    obtained_distribution, obtained_intervals = bootstrap_from_time_series(
        data_nest, "Nest", N=10, return_distribution=True
    )
    expected_distribution = [
        (1.8248880416251057, 2.100653559070486),
        (1.8089009429675704, 2.1287122021454903),
        (1.7575757577939, 2.2531034477499023),
        (1.8174934096625839, 2.1035558283175577),
        (1.810200672551555, 2.109785659442113),
        (1.8050173285456454, 2.1778260178683357),
        (1.8089009429675702, 2.1287122021454907),
        (1.8081803236719451, 2.119011965634283),
        (1.7885195974207062, 2.1707743395392596),
        (1.8098415460577582, 2.120238993094641),
    ]
    assert obtained_distribution == expected_distribution
    expected_intervals = [
        (1.7575757577939, 2.2531034477499023),
        (1.8089009429675702, 2.1287122021454907),
        (1.8248880416251057, 2.100653559070486),
    ]
    assert obtained_intervals == expected_intervals


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


data_test = np.arange(0, 10)
N_test = 5


def test_calculate_bootstrapped_mean():
    obtained_interval, obtained_latex_string = calculate_bootstrapped_mean(data_test, N=N_test)
    expected_latex_string = "5.0 (4.0 - 6.0)"
    expected_interval = np.array([4.19, 5.1, 6.19])
    np.testing.assert_almost_equal(obtained_interval, expected_interval, decimal=2)
    assert obtained_latex_string == expected_latex_string


def test_mean_bootstrapped():
    obtained_distribution = mean_bootstrapped(data_test, N=N_test)
    expected_distribution = [4.1, 5.0, 5.1, 6.2, 6.1]
    np.testing.assert_array_equal(obtained_distribution, expected_distribution)
    obtained_distribution = mean_bootstrapped(data_test)
    default_bootstrapping_size_sample = 2000
    assert len(obtained_distribution) == default_bootstrapping_size_sample
