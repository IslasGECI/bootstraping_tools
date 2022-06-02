import pandas as pd
from pandas.testing import assert_frame_equal
from bootstrapping_tools import (
    resample_data_by_blocks,
)


def test_resample_data_by_blocks():
    blocks_length = 2
    block_labels = [0, 0]
    block_size_2 = Tester_By_Size_Blocks(block_labels, blocks_length)
    data = pd.DataFrame({"a": [10, 20, 30], "b": [40, 60, 80]})
    expected = pd.DataFrame({"a": [10, 20, 10, 20], "b": [40, 60, 40, 60]})
    block_size_2.assert_resampled_by_blocks(data, expected)
    expected = pd.DataFrame({"a": [10, 20, 20, 30], "b": [40, 60, 60, 80]})
    block_size_2.block_labels = [0, 1]
    block_size_2.assert_resampled_by_blocks(data, expected)
    expected = pd.DataFrame({"a": [20, 30, 10, 20], "b": [60, 80, 40, 60]})
    block_size_2.block_labels = [1, 0]
    block_size_2.assert_resampled_by_blocks(data, expected)
    block_size_2.block_labels = [0, 0, 1]
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [4, 6, 8, 10, 12]})
    expected = pd.DataFrame({"a": [1, 2, 1, 2, 2, 3], "b": [4, 6, 4, 6, 6, 8]})
    block_size_2.assert_resampled_by_blocks(data, expected)
    block_size_2.block_labels = [0, 2, 3]
    expected = pd.DataFrame({"a": [1, 2, 3, 4, 4, 5], "b": [4, 6, 8, 10, 10, 12]})
    block_size_2.assert_resampled_by_blocks(data, expected)


def test_resample_data_by_blocks_of_size_3():
    blocks_length = 3
    block_labels = [0, 1]
    block_size_3 = Tester_By_Size_Blocks(block_labels, blocks_length)
    data = _data_from_columns([1, 2, 3, 4, 5], [4, 6, 8, 10, 12])
    expected = _data_from_columns([1, 2, 3, 2, 3, 4], [4, 6, 8, 6, 8, 10])
    block_size_3.assert_resampled_by_blocks(data, expected)


def assert_resampled_by_blocks(block_labels, data, expected, blocks_length):
    obtained = resample_data_by_blocks(data, block_labels, blocks_length)
    assert_frame_equal(expected.reset_index(drop=True), obtained.reset_index(drop=True))


class Tester_By_Size_Blocks:
    def __init__(self, block_labels, blocks_length):
        self.block_labels = block_labels
        self.blocks_length = blocks_length

    def assert_resampled_by_blocks(self, data, expected):
        obtained = resample_data_by_blocks(data, self.block_labels, self.blocks_length)
        assert_frame_equal(expected.reset_index(drop=True), obtained.reset_index(drop=True))


def _data_from_columns(column_a, column_b):
    return pd.DataFrame({"a": column_a, "b": column_b})
