import numpy as np


def random_resample_data_by_blocks(original_sample, blocks_length, rng):
    n_rows_original = len(original_sample)
    block_labels = get_labels(n_rows_original, blocks_length)
    number_of_blocks_to_choose = int(np.ceil(n_rows_original / blocks_length))
    block_labels = rng.choices(block_labels, k=number_of_blocks_to_choose)
    rows = _get_rows(block_labels, n_rows_original, blocks_length)
    resample = original_sample.iloc[rows, :].reset_index(drop=True)
    return resample


def _get_labels(n_rows_original, blocks_length):
    blocks_number = np.ceil(n_rows_original / blocks_length)
    block_labels = np.arange(blocks_number, dtype=int)
    return block_labels


def get_labels(n_rows_original, blocks_length):
    blocks_number = n_rows_original - blocks_length + 1
    block_labels = np.arange(blocks_number, dtype=int)
    return block_labels


def _get_rows(block_labels, n_rows_data, blocks_length):
    aux = np.arange(n_rows_data)
    rows = []
    for i in block_labels:
        rows.extend(aux[0 + i : blocks_length + i])
    return rows
