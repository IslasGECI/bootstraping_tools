from bootstrapping_tools import get_labels, resample_and_shift_data
import pandas as pd
import random


def test_distribution():
    gumu_data = pd.read_csv("tests/data/gumu_guadalupe_data.csv")
    blocks_length = 1
    seed = 99
    df_resample = resample_and_shift_data(gumu_data, seed, blocks_length)
    expected_nidos = gumu_data.Maxima_cantidad_nidos.to_list()
    obtained_nidos = df_resample.Maxima_cantidad_nidos.to_list()
    expected_temporadas = [i for i in range(len(gumu_data))]
    obtained_temporadas = df_resample.Temporada.to_list()
    assert all([obtained_nido in expected_nidos for obtained_nido in obtained_nidos])
    assert all(
        [obtained_temporada in expected_temporadas for obtained_temporada in obtained_temporadas]
    )

    blocks_length = 3
    seed = 7
    df_resample = resample_and_shift_data(gumu_data, seed, blocks_length)
    expected_nidos = [6.0, 40.0, 125.0, 2.0, 6.0, 40.0, 125.0, 195.0, 275.0]
    expected_temporadas = [1, 2, 3, 0, 1, 2, 3, 4, 5]
    obtained_nidos = df_resample.Maxima_cantidad_nidos.to_list()
    obtained_temporadas = df_resample.Temporada.to_list()
    assert obtained_nidos == expected_nidos
    assert obtained_temporadas == expected_temporadas


def test_get_labels():
    n_rows_original = 8
    blocks_length = 3
    obtained_labels = get_labels(n_rows_original, blocks_length)
    assert max(obtained_labels) == 5

    n_rows_original = 13
    blocks_length = 2
    obtained_labels = get_labels(n_rows_original, blocks_length)
    assert max(obtained_labels) == 11

    n_rows_original = 11
    blocks_length = 5
    obtained_labels = get_labels(n_rows_original, blocks_length)
    assert max(obtained_labels) == 6
