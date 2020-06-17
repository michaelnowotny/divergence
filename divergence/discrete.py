import numba
import numpy as np
import typing as tp


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


def discrete_entropy(sample: np.ndarray, log_fun: tp.Callable = np.log) -> float:
    """
    Approximate the entropy of a discrete distribution

                H(p) = - E_p[log(p)]

    from a sample.

    Parameters
    ----------
    sample: a sample from the discrete distribution
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    An approximation of the entropy of the discrete distribution from which the sample is drawn.

    """
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
    Construct two numpy arrays of frequencies for corresponding observations from sorted
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
                              log_fun: tp.Callable = np.log):
    """
    Approximate the relative entropy of the discrete distribution q relative to the discrete
    distribution p

                D_KL(p||q) = E_p [log(p/q)]

    from samples of these distributions.

    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.

    """
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
                           log_fun: tp.Callable = np.log):
    """
    Approximate the cross entropy of the discrete distribution q relative to the discrete
    distribution p

                H(p, q) = - E_p [log(q)]

    from samples of these distributions.

    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The cross entropy of the distribution q relative to the distribution p.

    """
    return discrete_relative_entropy(sample_p=sample_p,
                                     sample_q=sample_q,
                                     log_fun=log_fun) + \
           discrete_entropy(sample=sample_p,
                            log_fun=log_fun)


def discrete_jensen_shannon_divergence(sample_p: np.ndarray,
                                       sample_q: np.ndarray,
                                       log_fun: tp.Callable = np.log):
    """
    Approximate the Jensen-Shannon divergence between discrete distributions p and q

                JSD(p||q) = 0.5 * (D_KL(p||m) + D_KL(q||m)), with m = 0.5 * (p + q)

    from samples of these distributions.

    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The Jensen-Shannon divergence between distributions p and q.

    """
    m = np.hstack((sample_p, sample_q))
    D_PM = discrete_relative_entropy(sample_p=sample_p, sample_q=m, log_fun=log_fun)
    D_QM = discrete_relative_entropy(sample_p=sample_q, sample_q=m, log_fun=log_fun)

    return 0.5 * D_PM + 0.5 * D_QM
