# Your statistics are hiding things from you

Correlation between temperature and corn yield in the US Midwest: r = 0.08. Almost zero. A report using this number would conclude temperature has little effect on crop output.

That conclusion would be wrong.

[IMAGE: crop_yield_heat_cliff.png]

The left panel shows what's actually happening. Yields rise with warmth until 29°C, then fall off a cliff. Linear correlation cancels the rise against the fall and returns near-zero. Mutual information, which captures all dependence and not just linear, gives 0.74 nats. It sees the cliff. Correlation can't.

This comes from Divergence, an open-source Python package that collects 75 years' worth of information-theoretic measures into a single `pip install`. Here are five things it does.

---

## 1. Tell you which direction information flows

The S&P 500 and the Nikkei 225 are correlated (r = 0.24). Correlation is symmetric. It can't tell you who moves first.

Transfer entropy can. Thomas Schreiber developed it in 2000 to measure directed information flow in physical systems. Applied to daily stock returns from 2020 to 2023:

- TE(S&P → Nikkei) = 0.16 nats
- TE(Nikkei → S&P) = 0.04 nats

A 4:1 ratio. The S&P leads the Nikkei.

[IMAGE: stock_market_contagion.png]

```python
from divergence import transfer_entropy
te = transfer_entropy(source=sp_returns, target=nikkei_returns)
```

---

## 2. Measure how much the data taught you

Dennis Lindley proved in 1956 that the KL divergence from prior to posterior is the exact amount of information an experiment provides. This connects Bayesian inference to Shannon's information theory in one equation.

We applied this to the Nile River. Using 100 years of annual flow measurements and a change-point model, we asked: when did the river's flow change?

[IMAGE: information_gain.png]

The change-point year shows the highest information gain. The posterior concentrates at 1898, the year the British started building the Aswan Dam. We provided no prior knowledge about the dam. The data found it.

```python
from divergence import information_gain
ig = information_gain(idata)  # works with ArviZ, PyMC, Stan, NumPyro, emcee
```

---

## 3. Separate the uncertainty you can fix from the uncertainty you can't

Every prediction has two kinds of uncertainty. Aleatoric: irreducible noise in the system. Epistemic: uncertainty about the model that more data could reduce.

For the Nile, 99% of the prediction uncertainty is aleatoric. The river is noisy. More observations won't change that much.

For the Phillips Curve (unemployment vs inflation, 64 years of US data), the relationship keeps shifting, and the model parameters have wide credible intervals. A frequentist point estimate hides this. It treats the epistemic component as zero.

```python
from divergence import uncertainty_decomposition
ud = uncertainty_decomposition(idata, group='posterior_predictive')
```

---

## 4. Catch convergence failures that R-hat misses

$\hat{R}$, the standard MCMC convergence diagnostic, compares between-chain and within-chain variances. $\hat{R}$ far from 1 reliably flags non-convergence. But $\hat{R} \approx 1$ is necessary, not sufficient. It only checks the first two moments.

We built an example where $\hat{R} = 1.008$ (well under the 1.01 threshold) and ESS = 5,526, but two of four chains missed a secondary mode entirely.

[IMAGE: rhat_failure.png]

The left panel: overlaid chain histograms. The coral chains never found the bump near $\theta = 3$. The right panel: pairwise p-values from our distributional test. Cross-group values are 0.005. $\hat{R}$ said converged. The two-sample test said no.

```python
from divergence import chain_two_sample_test
result = chain_two_sample_test(idata, method='energy', n_permutations=200)
```

---

## 5. Handle 50,000 samples without blowing up

Energy distance, MMD, and kernel Stein discrepancy all need $O(n^2)$ pairwise comparisons. A naive implementation allocates an $n \times n$ matrix. At $n = 50{,}000$, that's 20 GB.

Divergence uses Numba JIT kernels that compute running sums instead of allocating matrices. Memory goes to $O(1)$. Runtimes drop 10-18x. Multicore parallelism is automatic.

| n | Energy distance | MMD | KSD |
|---|---|---|---|
| 5,000 | 21 ms | 136 ms | 114 ms |
| 20,000 | 276 ms | 1.9 s | 1.4 s |
| 50,000 | 1.4 s | 12 s | 8.6 s |

---

## The full list

83 functions. Shannon measures, f-divergences, Rényi family, energy/Wasserstein/MMD, kNN estimators, total correlation, transfer entropy, Fisher divergence, kernel Stein discrepancy (RBF + IMQ), Sinkhorn divergence, and a full suite of Bayesian MCMC diagnostics.

Discrete and continuous. Configurable units. Works with ArviZ, PyMC, Stan, NumPyro, PyJAGS, emcee. 345 tests.

```bash
pip install divergence
```

Seven notebooks with worked examples on real data:

github.com/michaelnowotny/divergence

Docs: michaelnowotny.github.io/divergence

---

*Built by Michael Nowotny. MIT license.*
