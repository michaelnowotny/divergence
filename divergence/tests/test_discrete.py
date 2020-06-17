import numpy as np
import pytest
import scipy as sp
import typing as tp

from divergence.discrete import (
    discrete_entropy,
    _construct_frequencies_for_one_sample,
    _construct_frequencies_for_two_samples,
    discrete_relative_entropy
)

multinomial_sample_q_1 = np.array([1, 2, 3, 2, 3, 3, 3, 2, 1, 1])
multinomial_sample_p_1 = np.array([2, 2, 3, 2, 3])
expected_frequencies_q_1 = np.array([0.3, 0.4])
expected_frequencies_p_1 = np.array([0.6, 0.4])

multinomial_sample_q_2 = np.array([1, 2, 3, 2, 3, 3, 3, 2, 1, 1])
multinomial_sample_p_2 = np.array([1, 2, 3, 2, 3])
expected_frequencies_q_2 = np.array([0.3, 0.3, 0.4])
expected_frequencies_p_2 = np.array([0.2, 0.4, 0.4])


def _get_base_from_log_fun(log_fun: tp.Callable):
    if log_fun is np.log:
        base = np.e
    elif log_fun is np.log2:
        base = 2
    elif log_fun is np.log10:
        base = 10
    else:
        raise ValueError('log_fun is not supported')

    return base


def discrete_entropy_scipy(sample: np.ndarray, log_fun: tp.Callable = np.log) -> float:
    base = _get_base_from_log_fun(log_fun)
    return sp.stats.entropy(_construct_frequencies_for_one_sample(sample), base=base)


@pytest.mark.parametrize("sample", (multinomial_sample_q_1,
                                    multinomial_sample_p_1,
                                    multinomial_sample_q_2,
                                    multinomial_sample_p_2))
@pytest.mark.parametrize("log_fun", (np.log, np.log2, np.log10))
def test_entropy(sample: np.ndarray, log_fun: tp.Callable):
    entropy_from_divergence = discrete_entropy(sample=sample, log_fun=log_fun)
    entropy_from_scipy = discrete_entropy_scipy(sample=sample, log_fun=log_fun)
    assert np.isclose(entropy_from_divergence, entropy_from_scipy)


@pytest.mark.parametrize("sample_p, sample_q, expected_frequencies_p, expected_frequencies_q",
                         [
                             (multinomial_sample_p_1, multinomial_sample_q_1, expected_frequencies_p_1, expected_frequencies_q_1),
                             (multinomial_sample_p_2, multinomial_sample_q_2, expected_frequencies_p_2, expected_frequencies_q_2)
                         ])
def test_construct_frequencies(sample_p: np.ndarray,
                               sample_q: np.ndarray,
                               expected_frequencies_p: np.ndarray,
                               expected_frequencies_q: np.ndarray):
    combined_sample = np.hstack((sample_p, sample_q))
    unique_combined = np.unique(combined_sample)

    unique_q, counts_q = np.unique(sample_q, return_counts=True)
    frequencies_q = counts_q / len(sample_q)
    # print(f'frequencies_q = {frequencies_q}')

    unique_p, counts_p = np.unique(sample_p, return_counts=True)
    frequencies_p = counts_p / len(sample_p)
    # print(f'frequencies_p = {frequencies_p}')

    combined_frequencies_p, combined_frequencies_q = \
        _construct_frequencies_for_two_samples(sorted_p_realizations=unique_p,
                                               sorted_q_realizations=unique_q,
                                               sorted_q_frequencies=frequencies_q,
                                               sorted_p_frequencies=frequencies_p,
                                               sorted_combined_realizations=unique_combined)

    assert np.allclose(combined_frequencies_p, expected_frequencies_p)
    assert np.allclose(combined_frequencies_q, expected_frequencies_q)


def test_construct_frequencies_error_q_zero_and_p_nonzero():
    sample_q = np.array([2, 2, 3, 2, 3, 3, 3, 2, 2, 2])
    sample_p = np.array([1, 2, 3, 2, 3])

    combined_sample = np.hstack((sample_p, sample_q))
    unique_combined = np.unique(combined_sample)

    unique_q, counts_q = np.unique(sample_q, return_counts=True)
    frequencies_q = counts_q / len(sample_q)
    # print(f'frequencies_q = {frequencies_q}')

    unique_p, counts_p = np.unique(sample_p, return_counts=True)
    frequencies_p = counts_p / len(sample_p)
    # print(f'frequencies_p = {frequencies_p}')

    with pytest.raises(ValueError):
        combined_frequencies_p, combined_frequencies_q = \
            _construct_frequencies_for_two_samples(sorted_p_realizations=unique_p,
                                                   sorted_q_realizations=unique_q,
                                                   sorted_q_frequencies=frequencies_q,
                                                   sorted_p_frequencies=frequencies_p,
                                                   sorted_combined_realizations=unique_combined)


def _discrete_relative_entropy_slow(sample_p: np.ndarray,
                                    sample_q: np.ndarray,
                                    log_fun: tp.Callable = np.log):
    combined_sample = np.hstack((sample_p, sample_q))
    unique_combined = np.unique(combined_sample)

    unique_q, counts_q = np.unique(sample_q, return_counts=True)
    frequencies_q = counts_q / len(sample_q)
    realization_to_frequency_dict_q = dict(zip(unique_q, frequencies_q))

    unique_p, counts_p = np.unique(sample_p, return_counts=True)
    frequencies_p = counts_p / len(sample_p)
    realization_to_frequency_dict_p = dict(zip(unique_p, frequencies_p))

    combined_frequencies_q = np.array([realization_to_frequency_dict_q.get(realization, 0.0)
                                       for realization
                                       in unique_combined])

    combined_frequencies_p = np.array([realization_to_frequency_dict_p.get(realization, 0.0)
                                       for realization
                                       in unique_combined])

    base = _get_base_from_log_fun(log_fun)
    # if log_fun is np.log:
    #     base = np.e
    # elif log_fun is np.log2:
    #     base = 2
    # elif log_fun is np.log10:
    #     base = 10
    # else:
    #     raise ValueError('log_fun is not supported')

    return sp.stats.entropy(pk=combined_frequencies_p, qk=combined_frequencies_q, base=base)


@pytest.mark.parametrize("sample_p, sample_q", ((multinomial_sample_p_1, multinomial_sample_q_1),
                                                (multinomial_sample_p_2, multinomial_sample_q_2)))
@pytest.mark.parametrize("log_fun", (np.log, np.log2, np.log10))
def test_compare_slow_and_fast_implementations_of_relative_entropy(sample_p: np.ndarray,
                                                                   sample_q: np.ndarray,
                                                                   log_fun: tp.Callable):
    relative_entropy_from_slow_calculation = \
        _discrete_relative_entropy_slow(sample_p=sample_p,
                                        sample_q=sample_q,
                                        log_fun=log_fun)

    relative_entropy_from_fast_calculation = \
        discrete_relative_entropy(sample_p=sample_p,
                                  sample_q=sample_q,
                                  log_fun=log_fun)

    assert np.isclose(relative_entropy_from_slow_calculation,
                      relative_entropy_from_fast_calculation)
