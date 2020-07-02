import numba
import numbers
import numpy as np
import typing as tp

from divergence.base import (
    _select_vectorized_log_fun_for_base,
    Logarithm
)


def _construct_counts_for_one_sample(sample: np.ndarray) -> np.ndarray:
    """
    Compute the count (i.e. number of occurrences) for each realization in the sample.
    The realizations in the argument `sample` do not need to be sorted. But the output counts will
    correspond to sorted realizations.

    Parameters
    ----------
    sample: a sample from the discrete distribution

    Returns
    -------
    Counts of realizations from a sample

    """
    _, counts = np.unique(sample, return_counts=True)
    return counts


def _construct_frequencies_for_one_sample(sample: np.ndarray) -> np.ndarray:
    """
    Compute the frequency (i.e. number of occurrences) for each realization in the sample.
    The realizations in the argument `sample` do not need to be sorted. But the output frequencies
    will correspond to sorted realizations.

    Parameters
    ----------
    sample: a sample from the discrete distribution

    Returns
    -------
    Frequencies of realizations from a sample

    """
    return _construct_counts_for_one_sample(sample) / len(sample)


def discrete_entropy(sample: np.ndarray,
                     base: float = np.e) -> float:
    """
    Approximate the entropy of a discrete distribution

                H(p) = - E_p[log(p)]

    from a sample.

    Parameters
    ----------
    sample: a sample from the discrete distribution
    base: the base of the logarithm used to control the units of measurement for the result

    Returns
    -------
    An approximation of the entropy of the discrete distribution from which the sample is drawn.

    """
    log_fun = _select_vectorized_log_fun_for_base(base)
    frequencies = _construct_frequencies_for_one_sample(sample)
    return - np.sum(frequencies * log_fun(frequencies))


@numba.njit
def _construct_frequencies_for_two_samples(sorted_p_realizations: np.ndarray,
                                           sorted_p_frequencies: np.ndarray,
                                           sorted_q_realizations: np.ndarray,
                                           sorted_q_frequencies: np.ndarray,
                                           sorted_combined_realizations: np.ndarray) \
        -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Construct two NumPy arrays of frequencies for corresponding observations from sorted
    realizations and frequencies from two samples. If a realization in the sample from q is not in
    the sample from p or has frequency zero then it is not included in either of the output
    frequency arrays.

    Parameters
    ----------
    sorted_p_realizations: NumPy array of unique realizations in the sample from p
    sorted_p_frequencies: The frequency of each realization in `sorted_p_realizations`
    sorted_q_realizations: NumPy array of unique realizations in the sample from q
    sorted_q_frequencies: The frequency of each realization in `sorted_q_realizations`
    sorted_combined_realizations: NumPy array of unique realizations in the samples from p and q
                                  combined

    Returns
    -------
    Two NumPy arraysof the same length with frequencies for corresponding observations that have
    positive weight in the sample from p.

    """
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

        if sorted_p_realizations[p_source_index] != realization:
            if sorted_q_realizations[q_source_index] == realization:
                q_source_index += 1
            continue

        if sorted_p_frequencies[p_source_index] == 0.0:
            p_source_index += 1
            if sorted_q_realizations[q_source_index] == realization:
                q_source_index += 1
            continue

        if sorted_q_realizations[q_source_index] != realization or \
           sorted_q_realizations[q_source_index] == 0.0:
            raise ValueError('q(x) is zero but p(x) is not')
            # if sorted_p_frequencies[p_source_index] != 0.0:  # we know that is true
            #     # if q(x) == 0 we must have p(x) == 0, which is not the case here
            #     raise ValueError('q(x) is zero but p(x) is not')
            # else:
            #     continue

        p_frequencies[p_target_index] = sorted_p_frequencies[p_source_index]
        q_frequencies[q_target_index] = sorted_q_frequencies[q_source_index]
        p_source_index += 1
        q_source_index += 1
        p_target_index += 1
        q_target_index += 1

    return p_frequencies[:p_target_index], q_frequencies[:q_target_index]


def discrete_relative_entropy(sample_p: np.ndarray,
                              sample_q: np.ndarray,
                              base: float = np.e):
    """
    Approximate the relative entropy of the discrete distribution q relative to the discrete
    distribution p

                D_KL(p||q) = E_p [log(p/q)]

    from samples of these distributions.

    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    base: the base of the logarithm used to control the units of measurement for the result

    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.

    """
    log_fun = _select_vectorized_log_fun_for_base(base)
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

    return np.sum(combined_frequencies_p * log_fun(combined_frequencies_p / combined_frequencies_q))


def discrete_cross_entropy(sample_p: np.ndarray,
                           sample_q: np.ndarray,
                           base: float = np.e):
    """
    Approximate the cross entropy of the discrete distribution q relative to the discrete
    distribution p

                H_q(p) = - E_p [log(q)]

    from samples of these distributions.

    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    base: the base of the logarithm used to control the units of measurement for the result

    Returns
    -------
    The cross entropy of the distribution q relative to the distribution p.

    """
    return discrete_relative_entropy(sample_p=sample_p,
                                     sample_q=sample_q,
                                     base=base) + \
           discrete_entropy(sample=sample_p,
                            base=base)


def discrete_jensen_shannon_divergence(sample_p: np.ndarray,
                                       sample_q: np.ndarray,
                                       base: float = np.e):
    """
    Approximate the Jensen-Shannon divergence between discrete distributions p and q

                JSD(p||q) = 0.5 * (D_KL(p||m) + D_KL(q||m)), with m = 0.5 * (p + q)

    from samples of these distributions.

    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    base: the base of the logarithm used to control the units of measurement for the result

    Returns
    -------
    The Jensen-Shannon divergence between distributions p and q.

    """
    m = np.hstack((sample_p, sample_q))
    D_PM = discrete_relative_entropy(sample_p=sample_p, sample_q=m, base=base)
    D_QM = discrete_relative_entropy(sample_p=sample_q, sample_q=m, base=base)

    return 0.5 * D_PM + 0.5 * D_QM


def _construct_unique_combinations_and_counts_from_two_samples(sample_x: np.ndarray,
                                                               sample_y: np.ndarray) \
        -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Construct an array of unique co-located combinations of sample_x and sample_y as well as an
    array of associated counts.

    Parameters
    ----------
    sample_x: a NumPy array of draws of variable x
    sample_y: a NumPy array of draws of variable y

    Returns
    -------
    a tuple of unique combinations of draws from x and y and associated counts
    """
    assert sample_x.ndim == 1
    assert sample_y.ndim == 1

    assert sample_x.shape == sample_y.shape

    n = len(sample_x)

    sample_x = sample_x.reshape((n, 1))
    sample_y = sample_y.reshape((n, 1))

    sample_xy = np.concatenate((sample_x, sample_y), axis=1)

    unique_combinations, counts = np.unique(sample_xy, axis=0, return_counts=True)

    return unique_combinations, counts


@numba.njit
def _get_index_for_combination(combination: np.ndarray,
                               unique_combinations: np.ndarray) -> int:
    """
    Returns the row index of a 2 element array in a nx2 dimensional array. Returns -1 if the
    requested array is not in the search array.

    Parameters
    ----------
    combination: an array whose position of first occurence is to be found
    unique_combinations: an array which is to be searched

    Returns
    -------
    the row index of the combination
    """
    for i in range(unique_combinations.shape[0]):
        if np.all(unique_combinations[i, :] == combination):
            return i

    return -1


@numba.njit
def _get_count_for_combination(combination: np.ndarray,
                               unique_combinations: np.ndarray,
                               counts: np.ndarray) -> int:
    """
    Given a 2x1 combination and arrays of unique combinations and associated counts, return the
    count of the combination.

    Parameters
    ----------
    combination: a 2 element array whose count is to be determined
    unique_combinations: a 2xn array of unique combinations
    counts: the count associated with the unique combinations

    Returns
    -------
    the count of the combination
    """

    return counts[_get_index_for_combination(combination=combination,
                                             unique_combinations=unique_combinations)]


@numba.njit
def _get_index_of_value_in_1d_array(value: numbers.Number,
                                    array: np.ndarray) -> int:
    """
    Returns the index of a value in an array and returns -1 if the array does not contain the value.
    Parameters
    ----------
    value: a number
    array: a one-dimensional NumPy array

    Returns
    -------
    the index of the value in the array
    """
    for i in range(len(array)):
        if value == array[i]:
            return i

    return -1


@numba.njit
def _get_count_for_value(value: numbers.Number,
                         unique_values: np.ndarray,
                         counts: np.ndarray) -> int:
    """
    Given a value and arrays of unique values and associated counts, return the
    count of the value.

    Parameters
    ----------
    value: a number whose count is to be determined
    unique_values: a one-dimensional array of unique values
    counts: the count associated with each unique value

    Returns
    -------
    the count of the value
    """

    return counts[_get_index_of_value_in_1d_array(value, unique_values)]


@numba.njit
def _discrete_mutual_information_internal(n: int,
                                          unique_combinations_xy: np.ndarray,
                                          counts_xy: np.ndarray,
                                          unique_values_x: np.ndarray,
                                          counts_x: np.ndarray,
                                          unique_values_y: np.ndarray,
                                          counts_y: np.ndarray,
                                          base: float = np.e) -> float:
    """
    Compute mutual information of discrete random variables x and y from

    Parameters
    ----------
    n: sample size
    unique_combinations_xy: NumPy array with shape (number_of_combinations, 2) of unique
                            combinations of X and Y appearing in the sample
    counts_xy: the number of appearances of each unique combination in the sample
    unique_values_x: NumPy array of the unique values of x
    counts_x: number of appearances of each unique value of x
    unique_values_y: NumPy array of the unique values of y
    counts_y: number of appearances of each unique value of y

    Returns
    -------

    """

    logarithm = Logarithm(base)

    mutual_information = 0.0
    for i in range(counts_xy.shape[0]):
        x = unique_combinations_xy[i, 0]
        y = unique_combinations_xy[i, 1]
        joint_count = counts_xy[i]

        x_count = _get_count_for_value(value=x,
                                       unique_values=unique_values_x,
                                       counts=counts_x)

        y_count = _get_count_for_value(value=y,
                                       unique_values=unique_values_y,
                                       counts=counts_y)

        mutual_information += (joint_count / n) * logarithm.log(n * joint_count / (x_count * y_count))

    return mutual_information


def _check_dimensions_of_two_variable_sample(sample_x: np.ndarray,
                                             sample_y: np.ndarray) \
        -> tp.Tuple[np.ndarray, np.ndarray, int]:
    """
    Check that sample_x and sample_y have the same number of elements and make them vectors.

    Parameters
    ----------
    sample_x: a NumPy array of draws of variable x
    sample_y: a NumPy array of draws of variable y

    Returns
    -------

    """
    if sample_x.ndim > 1:
        raise ValueError('sample_x must be a one dimensional array')

    if sample_y.ndim > 1:
        raise ValueError('sample_y must be a one dimensional array')

    sample_x = sample_x.reshape((-1, ))
    sample_y = sample_y.reshape((-1, ))

    n = len(sample_x)

    if n != len(sample_y):
        raise ValueError('sample_x and sample_y must have the same length')

    return sample_x, sample_y, n


def discrete_mutual_information(sample_x: np.ndarray,
                                sample_y: np.ndarray,
                                base: float = np.e) -> float:
    """
    Approximate the mutual information of x and y

            I(X; Y) = D_KL(p_{x, y}|| p_x \otimes p_y) =
            E_{p_{x, y}} \left[ \log \left( \frac{p_{x, y} (x, y)}{p_x(x) p_y(y)} \right) \right]

    from a sample of both distributions.

    Parameters
    ----------
    sample_x: a NumPy array of draws of variable x
    sample_y: a NumPy array of draws of variable y

    Returns
    -------
    The mutual information of x and y.
    """
    sample_x, sample_y, n = _check_dimensions_of_two_variable_sample(sample_x, sample_y)

    unique_combinations_xy, counts_xy = \
        _construct_unique_combinations_and_counts_from_two_samples(sample_x, sample_y)

    unique_values_x, counts_x = np.unique(sample_x, return_counts=True)
    unique_values_y, counts_y = np.unique(sample_y, return_counts=True)

    return _discrete_mutual_information_internal(n=n,
                                                 unique_combinations_xy=unique_combinations_xy,
                                                 counts_xy=counts_xy,
                                                 unique_values_x=unique_values_x,
                                                 counts_x=counts_x,
                                                 unique_values_y=unique_values_y,
                                                 counts_y=counts_y,
                                                 base=base)


def discrete_joint_entropy(sample_x: np.ndarray,
                           sample_y: np.ndarray,
                           base: float = np.e) -> float:
    """
    Approximate the joint entropy of x and y

       H(X, Y) = - E_{p_{x, y}} \left[ \log p_{x, y} (x, y) \right]

    from a sample of both distributions.

    Parameters
    ----------
    sample_x: a NumPy array of draws of variable x
    sample_y: a NumPy array of draws of variable y

    Returns
    -------
    The joint entropy between of x and y
    """

    log_fun = _select_vectorized_log_fun_for_base(base)
    sample_x, sample_y, n = _check_dimensions_of_two_variable_sample(sample_x, sample_y)

    unique_combinations_xy, counts_xy = \
        _construct_unique_combinations_and_counts_from_two_samples(sample_x, sample_y)

    joint_frequency = (1.0 / n) * counts_xy

    return - np.sum(joint_frequency * log_fun(joint_frequency))


@numba.njit
def _get_conditional_frequency_of_y_given_x(n: int,
                                            x: numbers.Number,
                                            y: numbers.Number,
                                            sample_x: np.ndarray,
                                            sample_y: np.ndarray) -> float:
    """
    Given a sample of two variables X and Y, and specific values of these variables x and y,
    determine the conditional frequency of Y=y given that X=x.

    Parameters
    ----------
    n: sample size
    x: value of x
    y: value of y
    sample_x: NumPy array containing the x-variable of the sample
    sample_y: NumPy array containing the y-variable of the sample

    Returns
    -------
    the conditional frequency of Y=y given that X=x
    """
    count_x = 0.0
    count_x_and_y = 0.0
    for i in range(n):
        if sample_x[i] == x:
            count_x += 1
            if sample_y[i] == y:
                count_x_and_y += 1

    if count_x == 0:
        raise ValueError('x value is not present in the sample')
    else:
        return count_x_and_y / count_x


@numba.njit
def _discrete_conditional_entropy_of_y_given_x_internal(n: int,
                                                        unique_combinations_xy: np.ndarray,
                                                        counts_xy: np.ndarray,
                                                        sample_x: np.ndarray,
                                                        sample_y: np.ndarray,
                                                        base: float = np.e) -> float:
    """
    Compute conditional entropy of discrete random variables X and Y from NumPy arrays of samples of
    these random variables. This function relies on pre-computed unique combinations of both
    variables and associated counts.

    Parameters
    ----------
    n: sample size
    unique_combinations_xy: NumPy array with shape (number_of_combinations, 2) of unique
                            combinations of X and Y appearing in the sample
    counts_xy: the number of appearances of each unique combination in the sample
    sample_x: NumPy array containing the x-variable of the sample
    sample_y: NumPy array containing the y-variable of the sample

    Returns
    -------
    The conditional entropy from a sample of discrete random variables
    """

    logarithm = Logarithm(base)

    conditional_entropy = 0.0
    for i in range(len(counts_xy)):
        x = unique_combinations_xy[i, 0]
        y = unique_combinations_xy[i, 1]

        conditional_frequency_of_y_given_x = \
            _get_conditional_frequency_of_y_given_x(n=n,
                                                    x=x,
                                                    y=y,
                                                    sample_x=sample_x,
                                                    sample_y=sample_y)
        conditional_entropy -= counts_xy[i] * logarithm.log(conditional_frequency_of_y_given_x) / n

    return conditional_entropy


def discrete_conditional_entropy_of_y_given_x(sample_x: np.ndarray,
                                              sample_y: np.ndarray,
                                              base: float = np.e) -> float:
    """
    Approximate the conditional entropy of y given x

        H(Y|X) = - E_{p_{x, y}} \left[ \log \frac{p_{x, y} (x, y)}{p_x(x)} \right]

    from a sample of both distributions.

    Parameters
    ----------
    sample_x: a NumPy array of draws of variable x
    sample_y: a NumPy array of draws of variable y

    Returns
    -------
    The conditional entropy between of y given x.
    """
    sample_x, sample_y, n = _check_dimensions_of_two_variable_sample(sample_x, sample_y)

    unique_combinations_xy, counts_xy = \
        _construct_unique_combinations_and_counts_from_two_samples(sample_x, sample_y)

    return _discrete_conditional_entropy_of_y_given_x_internal(
                n=n,
                unique_combinations_xy=unique_combinations_xy,
                counts_xy=counts_xy,
                sample_x=sample_x,
                sample_y=sample_y,
                base=base)
