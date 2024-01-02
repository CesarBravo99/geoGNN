import numpy as np


def sampling(src_nodes, sample_num, neighbor_table):
    """
    Arguments:
        src_nodes {list, ndarray}
        sample_num {int}
        neighbor_table {dict}
    
    Returns:
        np.ndarray
    """
    results = []
    for sid in src_nodes:
        res = np.random.choice(neighbor_table[sid], size=(sample_num, ))
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """
    Arguments:
        src_nodes {list, np.ndarray}
        sample_nums {list of int}
        neighbor_table {dict}
    
    Returns:
        [list of ndarray]
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result
