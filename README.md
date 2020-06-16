# Divergence
Divergence is a Python package to compute statistical measures of entropy and divergence from probability distributions and samples.

The following functionality is provided:
* (Information) Entropy
* Cross Entropy 
* Relative Entropy or Kullback-Leibler Divergence
* Jensen-Shannon Divergence

The units in which these entropy and divergence measures are calculated can be specified by the user. 
This is achieved by the argument `log_fun`, which accepts a function that calculates the logarithm with respect to a particular base. 
The following units can be realized by the corresponding choice of the argument `log_fun` in the entropy and divergence calculation functions:
* bits: base 2 via `np.log2`
* nats: base e via `np.log`
* dits: base 10 via `np.log10`

In a Bayesian context, relative entropy can be used as a measure of the information gained by moving 
from a prior distribution `q` to a posterior distribution `p`.


## Installation

<pre>
    pip install divergence
</pre>

## Examples
See the Jupyter notebook [Divergence](notebooks/Divergence.ipynb).