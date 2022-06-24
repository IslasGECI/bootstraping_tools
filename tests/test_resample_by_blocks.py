import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from bootstrapping_tools import resample_data_by_blocks, random_resample_data_by_blocks


def test_resample_data_by_blocks():
    blocks_length = 1
    block_size_1 = Tester_By_Size_Blocks(blocks_length)
    block_size_1.set_expected([10, 20, 30, 40, 50], [600, 700, 800, 900, 1000])
    block_size_1.assert_resampled_by_blocks()
    blocks_length = 3
    block_size_3 = Tester_By_Size_Blocks(blocks_length)
    block_size_3.set_expected([10, 20, 30, 20, 30, 40], [600, 700, 800, 700, 800, 900])
    block_size_3.assert_resampled_by_blocks()
    blocks_length = 4
    block_size_4 = Tester_By_Size_Blocks(blocks_length)
    block_size_4.set_expected(
        [10, 20, 30, 40, 20, 30, 40, 50], [600, 700, 800, 900, 700, 800, 900, 1000]
    )
    block_size_4.assert_resampled_by_blocks()


def test_random_resample_data_by_blocks():
    blocks_length = 4
    block_size_4 = Tester_By_Size_Blocks(blocks_length)
    block_size_4.set_expected(
        [10, 20, 30, 40, 20, 30, 40, 50], [600, 700, 800, 900, 700, 800, 900, 1000]
    )
    block_size_4.assert_random_resampled_by_blocks()


def test_length_block_labels():
    blocks_length = 5
    lenght_block_labels = 1
    block_size_4 = Tester_By_Size_Blocks(blocks_length)
    block_size_4.set_expected([10, 20, 30, 40, 50], [600, 700, 800, 900, 1000])
    n_rows_data = len(block_size_4.data)
    blocks_number = np.ceil(n_rows_data / blocks_length)
    assert lenght_block_labels <= blocks_number


class Tester_By_Size_Blocks:
    def __init__(self, blocks_length):
        self.blocks_length = blocks_length
        self.expected = None
        self.data = None
        self._set_data()

    def assert_resampled_by_blocks(self):
        obtained = resample_data_by_blocks(self.data, self.blocks_length)
        assert_frame_equal(self.expected.reset_index(drop=True), obtained.reset_index(drop=True))

    def assert_random_resampled_by_blocks(self):
        obtained = random_resample_data_by_blocks(self.data, self.blocks_length)
        assert_frame_equal(self.expected.reset_index(drop=True), obtained.reset_index(drop=True))

    def set_expected(self, column_a, column_b):
        self.expected = self._data_from_columns(column_a, column_b)

    def _set_data(self):
        self.data = self._data_from_columns(np.arange(10, 60, 10), np.arange(600, 1100, 100))

    def _data_from_columns(self, column_a, column_b):
        return pd.DataFrame({"a": column_a, "b": column_b})
