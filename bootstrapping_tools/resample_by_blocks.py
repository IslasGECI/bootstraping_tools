import numpy as np


def resample_data_by_blocks(original_sample, block_numbers, blocks_length):
    n_rows_original = len(original_sample)
    rows = get_rows(block_numbers, n_rows_original, blocks_length)
    resample = original_sample.loc[rows, :]
    return resample


def get_rows(block_numbers, n_rows_data, blocks_length):
    aux = np.arange(n_rows_data)
    rows = []
    for i in block_numbers:
        rows.extend(aux[0 + i : blocks_length + i])
    return rows
