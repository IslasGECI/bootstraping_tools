import numpy as np


def resample_data_by_blocks(original_sample, block_labels, blocks_length):
    n_rows_original = len(original_sample)
    rows = _get_rows(block_labels, n_rows_original, blocks_length)
    resample = original_sample.loc[rows, :]
    return resample


def _get_rows(block_labels, n_rows_data, blocks_length):
    blocks_number = np.ceil(n_rows_data/blocks_length)
    lenght_block_labels = len(block_labels)
    assert lenght_block_labels <= blocks_number
    aux = np.arange(n_rows_data)
    rows = []
    for i in block_labels:
        rows.extend(aux[0 + i : blocks_length + i])
    return rows
