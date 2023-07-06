import numpy as np


def random_resample_data_by_blocks(original_sample, blocks_length, rng):
    n_rows_original = len(original_sample)
    block_labels = get_labels(n_rows_original)
    number_of_blocks_to_choose = int(np.ceil(n_rows_original / blocks_length))
    choosen_block_labels = rng.choices(block_labels, k=number_of_blocks_to_choose)
    rows = get_rows(choosen_block_labels, n_rows_original, blocks_length)
    resample = original_sample.iloc[rows, :].reset_index(drop=True)
    return resample


def get_labels(n_rows_original):
    return np.arange(n_rows_original, dtype=int)


def get_rows(block_labels, n_rows_data, blocks_length):
    aux = [i for i in range(n_rows_data)]
    aux_cycle = [i for i in range(blocks_length - 1)]
    aux.extend(aux_cycle)
    rows = []
    for i in block_labels:
        rows.extend(aux[i : blocks_length + i])
    return rows
