import numbers
import numpy as np
import pytest
import scipy as sp
import typing as tp

from divergence.base import _select_vectorized_log_fun_for_base

from divergence.discrete import (
    discrete_entropy,
    _construct_frequencies_for_one_sample,
    _construct_frequencies_for_two_samples,
    discrete_relative_entropy,
    _construct_unique_combinations_and_counts_from_two_samples,
    _get_index_for_combination,
    _get_count_for_combination,
    _get_index_of_value_in_1d_array,
    _get_count_for_value,
    discrete_mutual_information,
    discrete_joint_entropy,
    discrete_conditional_entropy_of_y_given_x
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
@pytest.mark.parametrize("base", (np.e, 2.0, 10.0))
def test_entropy(sample: np.ndarray, base: float):
    log_fun = _select_vectorized_log_fun_for_base(base)
    entropy_from_divergence = discrete_entropy(sample=sample, base=base)
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

    unique_p, counts_p = np.unique(sample_p, return_counts=True)
    frequencies_p = counts_p / len(sample_p)

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

    unique_p, counts_p = np.unique(sample_p, return_counts=True)
    frequencies_p = counts_p / len(sample_p)

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
@pytest.mark.parametrize("base", (np.e, 2.0, 10.0))
def test_compare_slow_and_fast_implementations_of_relative_entropy(sample_p: np.ndarray,
                                                                   sample_q: np.ndarray,
                                                                   base: float):
    log_fun = _select_vectorized_log_fun_for_base(base)

    relative_entropy_from_slow_calculation = \
        _discrete_relative_entropy_slow(sample_p=sample_p,
                                        sample_q=sample_q,
                                        log_fun=log_fun)

    relative_entropy_from_fast_calculation = \
        discrete_relative_entropy(sample_p=sample_p,
                                  sample_q=sample_q,
                                  base=base)

    assert np.isclose(relative_entropy_from_slow_calculation,
                      relative_entropy_from_fast_calculation)


@pytest.fixture
def sample_x() -> np.ndarray:
    return np.array([1, 1, 3, 1, 2, 3])


@pytest.fixture
def sample_y() -> np.ndarray:
    return np.array([1, 1, 1, 3, 2, 1])


def test_construct_unique_combinations_and_counts_from_two_samples(sample_x, sample_y):
    unique_combinations, counts = \
        _construct_unique_combinations_and_counts_from_two_samples(sample_x, sample_y)

    print('unique combinations:')
    print(unique_combinations)

    print('counts')
    print(counts)

    assert np.all(unique_combinations == np.array([[1, 1], [1, 3], [2, 2], [3, 1]]))
    assert np.all(counts == np.array([2, 1, 1, 2]))


@pytest.mark.parametrize('combination, index', [(np.array([1, 1]), 0),
                                                (np.array([1, 3]), 1),
                                                (np.array([2, 2]), 2),
                                                (np.array([3, 1]), 3)])
def test_get_index_for_combination(combination: np.ndarray,
                                   index: int,
                                   sample_x: np.ndarray,
                                   sample_y: np.ndarray):
    unique_combinations, counts = \
        _construct_unique_combinations_and_counts_from_two_samples(sample_x, sample_y)

    assert index == _get_index_for_combination(combination=combination,
                                               unique_combinations=unique_combinations)


@pytest.mark.parametrize('combination, count', [(np.array([1, 1]), 2),
                                                (np.array([1, 3]), 1),
                                                (np.array([2, 2]), 1),
                                                (np.array([3, 1]), 2)])
def test_get_count_for_combination(combination: np.ndarray,
                                   count: int,
                                   sample_x: np.ndarray,
                                   sample_y: np.ndarray):
    unique_combinations, counts = \
        _construct_unique_combinations_and_counts_from_two_samples(sample_x, sample_y)

    assert count == _get_count_for_combination(combination=combination,
                                               unique_combinations=unique_combinations,
                                               counts=counts)


@pytest.mark.parametrize('value, index', [(1, 0), (2, 1), (3, 2)])
def test_get_index_for_value(value: numbers.Number,
                             index: int,
                             sample_x: np.ndarray):
    unique_values = np.unique(sample_x)

    assert index == _get_index_of_value_in_1d_array(value, unique_values)


@pytest.mark.parametrize('value, count', [(1, 3), (2, 1), (3, 2)])
def test_get_count_for_value(value: numbers.Number,
                             count: int,
                             sample_x: np.ndarray):
    unique_values, counts = np.unique(sample_x, return_counts=True)

    assert count == _get_count_for_value(value,
                                         unique_values=unique_values,
                                         counts=counts)


@pytest.mark.parametrize('sample', [np.array([1, 1, 3, 1, 2, 3]),
                                    np.array([1, 1, 1, 3, 2, 1]),
                                    np.array([1, 1, 1, 1, 1, 1])])
def test_compare_mutual_information_of_self_with_entropy(sample):
    assert discrete_entropy(sample) == discrete_mutual_information(sample, sample)


@pytest.mark.parametrize('sample_x, sample_y',
                         [(np.array([1, 1, 3, 1, 2, 3]), np.array([1, 1, 1, 3, 2, 1])),
                          (np.array([1, 1, 1, 1, 1, 1]), np.array([2, 2, 2, 2, 2, 2]))])
def test_symmetry_of_mutual_information(sample_x, sample_y):
    assert discrete_mutual_information(sample_x, sample_y) == \
           discrete_mutual_information(sample_y, sample_x)


def test_discrete_conditional_entropy(sample_x: np.ndarray, sample_y: np.ndarray):
    joint_entropy = discrete_joint_entropy(sample_x=sample_x, sample_y=sample_y)
    entropy_x = discrete_entropy(sample_x)
    entropy_y = discrete_entropy(sample_y)
    conditional_entropy_of_y_given_x = \
        discrete_conditional_entropy_of_y_given_x(sample_x=sample_x,
                                                  sample_y=sample_y)

    conditional_entropy_of_x_given_y = \
        discrete_conditional_entropy_of_y_given_x(sample_x=sample_y,
                                                  sample_y=sample_x)

    assert np.isclose(entropy_x - conditional_entropy_of_x_given_y,
                      entropy_y - conditional_entropy_of_y_given_x)

    assert np.isclose(joint_entropy, entropy_x + conditional_entropy_of_y_given_x)
    assert np.isclose(joint_entropy, entropy_y + conditional_entropy_of_x_given_y)


def test_discrete_mutual_information_and_conditional_entropy(sample_x: np.ndarray,
                                                             sample_y: np.ndarray):
    mutual_information = discrete_mutual_information(sample_x=sample_x, sample_y=sample_y)

    entropy_x = discrete_entropy(sample_x)
    entropy_y = discrete_entropy(sample_y)
    conditional_entropy_of_y_given_x = \
        discrete_conditional_entropy_of_y_given_x(sample_x=sample_x,
                                                  sample_y=sample_y)

    conditional_entropy_of_x_given_y = \
        discrete_conditional_entropy_of_y_given_x(sample_x=sample_y,
                                                  sample_y=sample_x)

    assert np.isclose(mutual_information, entropy_x - conditional_entropy_of_x_given_y)
    assert np.isclose(mutual_information, entropy_y - conditional_entropy_of_y_given_x)
