import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from bootstrapping_tools import xxrandom_resample_data_by_blocks, xxresample_and_shift_data
import random


def test_resample_and_shift_data():
    blocks_length = 2
    block_size_2 = Size_blocks_tester(blocks_length)
    block_size_2.set_expected([20, 30, 20, 30, 0, 10], [800, 900, 800, 900, 600, 700])
    random_seed = 2
    block_size_2.assert_resample_and_shift_data(random_seed)


def test_resample_and_shift_data_block_length_4():
    blocks_length = 4
    block_size_4 = Size_blocks_tester(blocks_length)
    block_size_4.set_expected(
        [10, 20, 30, 40, 10, 20, 30, 40], [700, 800, 900, 1000, 700, 800, 900, 1000]
    )
    random_seed = 2
    block_size_4.assert_resample_and_shift_data(random_seed)


def test_random_resample_data_by_blocks_blocks_length_2():
    blocks_length = 2
    block_size_2 = Size_blocks_tester(blocks_length)
    block_size_2.set_expected([40, 50, 40, 50, 10, 20], [900, 1000, 900, 1000, 600, 700])
    random_seed = 2
    rng = random.Random(random_seed)
    block_size_2.assert_random_resampled_by_blocks(rng)


def test_random_resample_data_by_blocks():
    blocks_length = 4
    block_size_4 = Size_blocks_tester(blocks_length)
    block_size_4.set_expected(
        [20, 30, 40, 50, 20, 30, 40, 50], [700, 800, 900, 1000, 700, 800, 900, 1000]
    )
    random_seed = 2
    rng = random.Random(random_seed)
    block_size_4.assert_random_resampled_by_blocks(rng)


def test_length_block_labels():
    blocks_length = 5
    lenght_block_labels = 1
    block_size_5 = Size_blocks_tester(blocks_length)
    block_size_5.set_expected([10, 20, 30, 40, 50], [600, 700, 800, 900, 1000])
    n_rows_data = len(block_size_5.data)
    blocks_number = np.ceil(n_rows_data / blocks_length)
    assert lenght_block_labels <= blocks_number

    blocks_length = 2
    lenght_block_labels = 3
    block_size_5 = Size_blocks_tester(blocks_length)
    n_rows_data = len(block_size_5.data)
    blocks_number = np.ceil(n_rows_data / blocks_length)
    assert lenght_block_labels <= blocks_number


class Size_blocks_tester:
    def __init__(self, blocks_length):
        self.blocks_length = blocks_length
        self.expected = None
        self.data = None
        self._set_data()

    def assert_random_resampled_by_blocks(self, rng):
        obtained = xxrandom_resample_data_by_blocks(self.data, self.blocks_length, rng)
        assert_frame_equal(self.expected, obtained)

    def assert_resample_and_shift_data(self, seed):
        obtained = xxresample_and_shift_data(self.data, seed, self.blocks_length)
        assert_frame_equal(self.expected, obtained)

    def set_expected(self, column_a, column_b):
        self.expected = self._data_from_columns(column_a, column_b).reset_index(drop=True)

    def _set_data(self):
        self.data = self._data_from_columns(np.arange(10, 60, 10), np.arange(600, 1100, 100))

    def _data_from_columns(self, column_a, column_b):
        return pd.DataFrame({"b": column_b, "Temporada": column_a})
