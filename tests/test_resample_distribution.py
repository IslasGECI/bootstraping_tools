from bootstrapping_tools import (
    get_labels,
    resample_and_shift_data,
    xxget_labels,
    get_rows,
)

from collections import Counter
import numpy as np
import pandas as pd
import random
import math


def test_proportions():
    N = 10000
    n_rows_data = 8
    blocks_length = 3
    number_of_blocks_to_choose = 3
    block_labels = np.arange(n_rows_data)
    obtained_rows = []
    for i in range(N):
        choosen_block_labels = random.choices(block_labels, k=number_of_blocks_to_choose)
        rows = get_rows(choosen_block_labels, n_rows_data, blocks_length)
        obtained_rows.extend(rows)
    ocurrences = Counter(obtained_rows)
    print(np.std(obtained_rows) ** 2)
    print(ocurrences)
    obtained_mean = np.mean(obtained_rows)
    print(obtained_mean)
    expected_mean = 0.5 * (n_rows_data - 1)
    assert math.isclose(obtained_mean, expected_mean, rel_tol=1e-2)


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
    expected_nidos = [40.0, 125.0, 195.0, 6.0, 40.0, 125.0, 275.0, 399.0, 686.0]
    expected_temporadas = [2, 3, 4, 1, 2, 3, 5, 6, 7]
    obtained_nidos = df_resample.Maxima_cantidad_nidos.to_list()
    obtained_temporadas = df_resample.Temporada.to_list()
    assert obtained_nidos == expected_nidos
    assert obtained_temporadas == expected_temporadas


def test_get_labels():
    n_rows_original = 8
    blocks_length = 3
    obtained_labels = xxget_labels(n_rows_original)
    assert max(obtained_labels) == 7

    n_rows_original = 13
    blocks_length = 2
    obtained_labels = xxget_labels(n_rows_original)
    assert max(obtained_labels) == 12


def tests_get_rows():
    blocks_length = 3
    n_rows_data = 8
    block_labels = np.arange(8, dtype=int)
    obtained_rows = get_rows(block_labels, n_rows_data, blocks_length)
    expected_rows = [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 0, 7, 0, 1]
    assert obtained_rows == expected_rows

    blocks_length = 3
    block_labels = np.array([4, 6, 2])
    obtained_rows = get_rows(block_labels, n_rows_data, blocks_length)
    expected_rows = [4, 5, 6, 6, 7, 0, 2, 3, 4]
    assert obtained_rows == expected_rows
