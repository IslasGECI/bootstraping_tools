from numpy.testing import assert_array_almost_equal
import pandas as pd
import pytest
import json
import os
from bootstrapping_tools import Bootstrap_from_time_series_parametrizer, LambdasBootstrapper


nidos_array = [
    {"Temporada": 2014, "Maxima_cantidad_nidos": 283},
    {"Temporada": 2015, "Maxima_cantidad_nidos": 126},
    {"Temporada": 2016, "Maxima_cantidad_nidos": 395},
    {"Temporada": 2017, "Maxima_cantidad_nidos": 344},
    {"Temporada": 2018, "Maxima_cantidad_nidos": 921},
    {"Temporada": 2019, "Maxima_cantidad_nidos": 847},
]


def get_df(file_path):
    with open(file_path) as file:
        df = pd.read_csv(file)
    return df


df = get_df("tests/data/dcco_laal_gumu_burrows_data.csv")
laal = df[df["Nombre_en_ingles"] == "Laysan Albatross"]
parametrizer = Bootstrap_from_time_series_parametrizer(blocks_length=2, N=100)
parametrizer.set_data(laal)
bootstraper = LambdasBootstrapper(parametrizer)


def test_save_intervals():
    output_path = "tests/data/laal_intervals.json"
    bootstraper.save_intervals(output_path)
    with open(output_path) as json_file:
        obtained_json = json.load(json_file)
    obtained_fields = list(obtained_json.keys())
    expected_fields = [
        "intervals",
        "lambda_latex_interval",
        "p-values",
        "bootstrap_intermediate_distribution",
    ]
    assert obtained_fields == expected_fields
    obtained_values = list(obtained_json.values())
    expected_intervals = [[1.11653, 150.30929], [1.21173, 77.48159], [1.4272, 4.56072]]
    assert_array_almost_equal(obtained_values[0], expected_intervals, decimal=5)
    expected_latex_interval = "1.21 (1.12 - 1.43)"
    assert obtained_values[1] == expected_latex_interval
    obtained_p_minor_value = obtained_values[2][0]
    assert obtained_p_minor_value >= 0
    obtained_p_major_value = obtained_values[2][1]
    assert obtained_p_major_value <= 1
    obtained_min_lambda = min(obtained_values[3])
    assert obtained_min_lambda[0] >= expected_intervals[0][0]
    obtained_max_lambda = max(obtained_values[3])
    assert obtained_max_lambda[0] <= expected_intervals[2][0]
    os.remove(output_path)


def test_intervals_from_p_values_and_alpha():
    dcco = df[df["Nombre_en_ingles"] == "Double-crested Cormorant"]
    parametrizer = Bootstrap_from_time_series_parametrizer(blocks_length=2, N=100)
    parametrizer.set_data(dcco)
    assert parametrizer.parameters["alpha"] == 0.05
    bootstraper = LambdasBootstrapper(parametrizer)
    obtained_intervals = bootstraper.intervals
    obtained_len_intervals = len(obtained_intervals)
    expected_len_intervals = 3
    assert obtained_len_intervals == expected_len_intervals
    obtained_intervals_property = bootstraper.intervals
    expected_intervals = [
        (1.03555254967221, 56.72689275689199),
        (1.2199265239402008, 14.422599452094458),
        (9.496750484649498, 2.419173122070242e-07),
    ]
    assert_array_almost_equal(obtained_intervals_property, expected_intervals, decimal=5)


def test_generate_season_interval():
    datos_di = {"Temporada": [1, 2, 3, 4, 5], "Maxima_cantidad_nidos": [1, 1, 1, 1, 1]}
    df = pd.DataFrame(datos_di)
    parametrizer = Bootstrap_from_time_series_parametrizer(blocks_length=2, N=100)
    parametrizer.set_data(df)
    bootstraper = LambdasBootstrapper(parametrizer)
    expected_interval = "(1-5)"
    obtained_interval = bootstraper.generate_season_interval()
    assert expected_interval == obtained_interval


testdata = [
    (
        "tests/data/unsorted_seasons.csv",
        "2010-2021",
    ),
    (
        "tests/data/repeated_seasons.csv",
        "2010-2021",
    ),
    (
        "tests/data/one_season.csv",
        "2010",
    ),
]


@pytest.mark.parametrize("path,expected_seasons", testdata)
def test_get_monitored_seasons(path, expected_seasons):
    burrows_data_dataframe = get_df(path)
    parametrizer = Bootstrap_from_time_series_parametrizer(blocks_length=2, N=100)
    parametrizer.set_data(burrows_data_dataframe)
    bootstraper = LambdasBootstrapper(parametrizer)
    obtained_seasons = bootstraper.get_monitored_seasons()
    assert expected_seasons == obtained_seasons
