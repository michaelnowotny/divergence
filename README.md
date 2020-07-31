# Divergence
Divergence is a Python package to compute statistical measures of entropy and divergence from probability distributions and samples.

The following functionality is provided:
* (Information) Entropy [1], [2]
* Cross Entropy: [3]
* Relative Entropy or Kullback-Leibler (KL-) Divergence [4], [5]
* Jensen-Shannon Divergence [6]
* Joint Entropy [7]
* Conditional Entropy [8]
* Mutual Information [9]

The units in which these entropy and divergence measures are calculated can be specified by the user. 
This is achieved by setting the argument `base`, to `2.0`, `10.0`, or `np.e`. 

In a Bayesian context, relative entropy can be used as a measure of the information gained by moving 
from a prior distribution `q` to a posterior distribution `p`.

## Installation

<pre>
    pip install divergence
</pre>

## Examples
See the Jupyter notebook [Divergence](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Divergence.ipynb).

##References: 
#### [1] https://en.wikipedia.org/wiki/Entropy_(information_theory)
#### [2] Shannon, Claude Elwood (July 1948). "A Mathematical Theory of Communication". Bell System Technical Journal. 27 (3): 379–423
#### [3] https://en.wikipedia.org/wiki/Cross_entropy
#### [4] https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
#### [5] Kullback, S.; Leibler, R.A. (1951). "On information and sufficiency". Annals of Mathematical Statistics. 22 (1): 79–86
#### [6] https://en.wikipedia.org/wiki/Jensen–Shannon_divergence
#### [7] https://en.wikipedia.org/wiki/Joint_entropy
#### [8] https://en.wikipedia.org/wiki/Conditional_entropy
#### [9] https://en.wikipedia.org/wiki/Mutual_information
