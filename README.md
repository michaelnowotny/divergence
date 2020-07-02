# Divergence
Divergence is a Python package to compute statistical measures of entropy and divergence from probability distributions and samples.

The following functionality is provided:
* (Information) Entropy
* Cross Entropy 
* Relative Entropy or Kullback-Leibler (KL-) Divergence
* Jensen-Shannon Divergence
* Joint Entropy
* Conditional Entropy
* Mutual Information

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
