import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from bootstrapping_tools import (
    resample_data_by_blocks,
)


def test_resample_data_by_blocks():
    blocks_length = 2
    block_labels = [0, 0]
    block_size_2 = Tester_By_Size_Blocks(block_labels, blocks_length)
    block_size_2.set_expected([10, 20, 10, 20], [600, 700, 600, 700])
    block_size_2.assert_resampled_by_blocks()
    block_size_2.block_labels = [0, 1]
    block_size_2.set_expected([10, 20, 20, 30], [600, 700, 700, 800])
    block_size_2.assert_resampled_by_blocks()
    block_size_2.block_labels = [1, 0]
    block_size_2.set_expected([20, 30, 10, 20], [700, 800, 600, 700])
    block_size_2.assert_resampled_by_blocks()
    block_size_2.block_labels = [0, 0, 1]
    block_size_2.set_expected([10, 20, 10, 20, 20, 30], [600, 700, 600, 700, 700, 800])
    block_size_2.assert_resampled_by_blocks()
    block_size_2.block_labels = [0, 2, 3]
    block_size_2.set_expected([10, 20, 30, 40, 40, 50], [600, 700, 800, 900, 900, 1000])
    block_size_2.assert_resampled_by_blocks()


def test_resample_data_by_blocks_of_size_3():
    blocks_length = 3
    block_labels = [0, 1]
    block_size_3 = Tester_By_Size_Blocks(block_labels, blocks_length)
    block_size_3.set_expected([10, 20, 30, 20, 30, 40], [600, 700, 800, 700, 800, 900])
    block_size_3.assert_resampled_by_blocks()


class Tester_By_Size_Blocks:
    def __init__(self, block_labels, blocks_length):
        self.block_labels = block_labels
        self.blocks_length = blocks_length
        self.expected = None
        self.data = None
        self._set_data()

    def assert_resampled_by_blocks(self):
        obtained = resample_data_by_blocks(self.data, self.block_labels, self.blocks_length)
        assert_frame_equal(self.expected.reset_index(drop=True), obtained.reset_index(drop=True))

    def set_expected(self, column_a, column_b):
        self.expected = self._data_from_columns(column_a, column_b)

    def _set_data(self):
        self.data = self._data_from_columns(np.arange(10, 60, 10), np.arange(600, 1100, 100))

    def _data_from_columns(self, column_a, column_b):
        return pd.DataFrame({"a": column_a, "b": column_b})
