import numpy as np
import typing as tp


def _construct_counts_for_one_sample(sample: np.ndarray):
    _, counts = np.unique(sample, return_counts=True)
    return counts


def _construct_frequencies_for_one_sample(sample: np.ndarray):
    return _construct_counts_for_one_sample(sample) / len(sample)


def discrete_entropy(sample: np.ndarray, log_fun: tp.Callable = np.log):
    frequencies = _construct_frequencies_for_one_sample(sample)
    return - np.sum(frequencies * log_fun(frequencies))


def _construct_frequencies_for_two_samples(sorted_p_realizations: np.ndarray,
                                           sorted_p_frequencies: np.ndarray,
                                           sorted_q_realizations: np.ndarray,
                                           sorted_q_frequencies: np.ndarray,
                                           sorted_combined_realizations: np.ndarray):
    assert len(sorted_p_realizations) == len(sorted_p_frequencies)
    assert len(sorted_q_realizations) == len(sorted_q_frequencies)

    p_source_index = 0
    q_source_index = 0
    p_target_index = 0
    q_target_index = 0

    p_frequencies = np.zeros((len(sorted_p_realizations, )))
    q_frequencies = np.zeros((len(sorted_p_realizations, )))

    for combined_index in range(len(sorted_combined_realizations)):
        realization = sorted_combined_realizations[combined_index]

        print(f'combined_index = {combined_index}, realization = {realization}')
        if sorted_p_realizations[p_source_index] != realization:
            print(f'realization {realization} is not in p')
            if sorted_q_realizations[q_source_index] == realization:
                print(f'but realization {realization} is in q')
                q_source_index += 1
            continue

        if sorted_p_frequencies[p_source_index] == 0.0:
            p_source_index += 1
            if sorted_q_realizations[q_source_index] == realization:
                q_source_index += 1
            continue

        if sorted_q_realizations[q_source_index] != realization or sorted_q_realizations[
            q_source_index] == 0.0:
            if sorted_p_frequencies[p_source_index] != 0.0:  # we know that is true
                # if q(x) == 0 we must have p(x) == 0, which is not the case here
                raise ValueError('q(x) is zero but p(x) is not')
            else:
                continue

        p_frequencies[p_target_index] = sorted_p_frequencies[p_source_index]
        q_frequencies[q_target_index] = sorted_q_frequencies[q_source_index]
        p_source_index += 1
        q_source_index += 1
        p_target_index += 1
        q_target_index += 1

    return p_frequencies[:p_target_index], q_frequencies[:q_target_index]


def discrete_relative_entropy(sample_p: np.ndarray,
                              sample_q: np.ndarray,
                              log_fun: tp.Callable = np.log):
    combined_sample = np.hstack((sample_p, sample_q))
    unique_combined = np.unique(combined_sample)

    unique_q, counts_q = np.unique(sample_q, return_counts=True)
    frequencies_q = counts_q / len(sample_q)
    # print(f'unique_q = {unique_q}')

    unique_p, counts_p = np.unique(sample_p, return_counts=True)
    frequencies_p = counts_p / len(sample_p)
    # print(f'unique_p = {unique_p}')

    combined_frequencies_p, combined_frequencies_q = \
        _construct_frequencies_for_two_samples(sorted_p_realizations=unique_p,
                                               sorted_q_realizations=unique_q,
                                               sorted_q_frequencies=frequencies_q,
                                               sorted_p_frequencies=frequencies_p,
                                               sorted_combined_realizations=unique_combined)

    # print(f'combined_frequencies_p = {combined_frequencies_p}')
    # print(f'combined_frequencies_q = {combined_frequencies_q}')

    return np.sum(combined_frequencies_p * log_fun(combined_frequencies_p / combined_frequencies_q))