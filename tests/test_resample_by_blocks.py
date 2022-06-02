import pandas as pd
from pandas.testing import assert_frame_equal
from bootstrapping_tools import (
    resample_data_by_blocks,
    get_rows,
)


def test_resample_data_by_blocks():
    blocks_length = 2
    block_numbers = [0, 0]
    data = pd.DataFrame({"a": [10, 20, 30], "b": [40, 60, 80]})
    expected = pd.DataFrame({"a": [10, 20, 10, 20], "b": [40, 60, 40, 60]})
    assert_resampled_by_blocks(block_numbers, data, expected, blocks_length)
    expected = pd.DataFrame({"a": [10, 20, 20, 30], "b": [40, 60, 60, 80]})
    block_numbers = [0, 1]
    assert_resampled_by_blocks(block_numbers, data, expected, blocks_length)
    expected = pd.DataFrame({"a": [20, 30, 10, 20], "b": [60, 80, 40, 60]})
    block_numbers = [1, 0]
    assert_resampled_by_blocks(block_numbers, data, expected, blocks_length)
    block_numbers = [0, 0, 1]
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [4, 6, 8, 10, 12]})
    expected = pd.DataFrame({"a": [1, 2, 1, 2, 2, 3], "b": [4, 6, 4, 6, 6, 8]})
    assert_resampled_by_blocks(block_numbers, data, expected, blocks_length)
    block_numbers = [0, 2, 3]
    expected = pd.DataFrame({"a": [1, 2, 3, 4, 4, 5], "b": [4, 6, 8, 10, 10, 12]})
    assert_resampled_by_blocks(block_numbers, data, expected, blocks_length)


def test_resample_data_by_blocks_of_size_3():
    blocks_length = 3
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [4, 6, 8, 10, 12]})
    block_numbers = [0, 1]
    expected = pd.DataFrame({"a": [1, 2, 3, 2, 3, 4], "b": [4, 6, 8, 6, 8, 10]})
    assert_resampled_by_blocks(block_numbers, data, expected, blocks_length)


def assert_resampled_by_blocks(block_numbers, data, expected, blocks_length):
    obtained = resample_data_by_blocks(data, block_numbers, blocks_length)
    assert_frame_equal(expected.reset_index(drop=True), obtained.reset_index(drop=True))
