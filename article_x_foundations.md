# What does it mean to measure information?

You flip a coin. How much did you learn?

The answer is one bit. Not metaphorically — the unit is literally named after this. Binary digit. One coin flip, one bit of information.

Roll a six-sided die: 2.58 bits. Read a letter of English text: about 1.1 bits, because after you see "q" you already know what's coming next. The language is redundant, and redundancy means less surprise.

In 1948, a Bell Labs mathematician named Claude Shannon figured out how to put a number on this. He called the number entropy. His 38-page paper launched a field. Compression algorithms, error-correcting codes, cryptography, neuroscience, machine learning, the theory behind large language models — all of it traces back to Shannon's question: how do you measure surprise?

I spent a long time looking for a clean, practical way to use these ideas in Python. The measures existed — scattered across textbooks, academic papers, and half a dozen packages with incompatible APIs. So I collected them into one place. The package is called Divergence, it's open-source, and this article is about the ideas inside it.

---

## Entropy: the quantity that measures uncertainty

Shannon's entropy for a discrete distribution is:

H = -Σ p(x) log p(x)

$$H(X) = -\sum_{x} p(x) \log p(x)$$

Each outcome contributes its probability times the log of its probability. Rare events contribute more. Common events contribute less. The sum tells you how uncertain you are about what comes next.

A fair coin: H = 1 bit. A loaded coin (90/10): H = 0.47 bits. You're less uncertain about the loaded coin because you can mostly predict it.

Here's the part that trips people up. A completely random sequence — white noise, no structure, no patterns — has *maximum* entropy. A Shakespeare sonnet has low entropy because English is predictable. Random noise contains more information in the Shannon sense than anything meaningful you've ever read.

This isn't a flaw. It's the point. Shannon's "information" means surprise, not meaning. Structured data is partially predictable, so each observation teaches you less. Random data is completely unpredictable, so every observation is maximally informative.

```python
from divergence import entropy
h = entropy(samples)
```

[IMAGE: divergence/two_distributions.png]

---

## KL divergence: the penalty for using the wrong model

Suppose you're compressing data. Your data comes from distribution P, but you designed your compression code for distribution Q. How much space do you waste?

The answer is the Kullback-Leibler divergence, $D_{\text{KL}}(P \| Q)$. It measures the extra cost — in bits or nats — of encoding P using Q's code. If P and Q are identical, the cost is zero. If they differ, you pay.

(If you've seen "perplexity" in NLP papers: that's just exponentiated cross-entropy. A language model with cross-entropy of 5.6 bits has perplexity $2^{5.6} \approx 49$ — it's as confused as if it were choosing uniformly among 49 words at each position. Same idea, different scale.)

Solomon Kullback and Richard Leibler were cryptanalysts at the NSA in 1951, working on the mathematics of distinguishing one message source from another. Was this intercepted signal random noise, or did it contain a hidden pattern? Their divergence gives a precise answer: how much evidence does the data provide that P and Q are different?

Decades later, Bayesian statisticians realized that KL divergence is exactly the information gained by updating from a prior to a posterior. The cryptanalysts' wartime tool and the Bayesian's measure of learning turned out to be the same equation.

One thing to know: KL divergence is not symmetric. $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$. Encoding P with Q's code is a different problem from encoding Q with P's code. This asymmetry is both a feature and a limitation — it motivates everything that comes next.

---

## The family you didn't know existed

For decades, people treated KL divergence, chi-squared, total variation, and Hellinger distance as separate tools with separate histories. Karl Pearson introduced chi-squared in 1900 for goodness-of-fit testing. Ernst Hellinger defined his distance in 1909 for a problem in functional analysis. Total variation comes from measure theory. KL divergence from cryptography.

In 1963, a Hungarian mathematician named Imre Csiszár proved they're all the same thing.

Every one of these measures is an f-divergence:

$$D_f(P \| Q) = \sum_x q(x) \cdot f\!\left(\frac{p(x)}{q(x)}\right)$$

where $f$ is a convex function. Change $f$ and you get a different divergence. Set $f(t) = t \log t$ and you get KL. Set $f(t) = (t-1)^2$ and you get chi-squared. Set $f(t) = (\sqrt{t} - 1)^2$ and you get Hellinger. Set $f(t) = |t-1|/2$ and you get total variation.

Different functions of the same density ratio. Each one emphasizes different kinds of distributional mismatch. KL is sensitive to where P has mass and Q doesn't. Chi-squared amplifies large density ratios. Hellinger is gentler — more robust to outliers. Total variation measures the largest possible difference in probability assigned to any event.

[IMAGE: beyond_kl/generator_functions.png]

But the real surprise came in 1984. Noel Cressie and Timothy Read showed that you can parameterize the entire family with a single number $\lambda$:

$$\text{CR}_\lambda(P \| Q) = \frac{1}{\lambda(\lambda+1)} \sum_x p(x) \left[\left(\frac{p(x)}{q(x)}\right)^\lambda - 1\right]$$

Set $\lambda = -1$: reverse KL. Set $\lambda = -\tfrac{1}{2}$: Hellinger. Set $\lambda = 0$: KL divergence. Set $\lambda = \tfrac{2}{3}$: the value Cressie and Read themselves recommended for goodness-of-fit testing. Set $\lambda = 1$: Pearson chi-squared. One formula. One parameter. A smooth continuum connecting measures that had been treated as unrelated for the better part of a century.

[IMAGE: cressie_read_continuum.png]

That plot is one of my favorite things in all of statistics. Five labeled points on a smooth curve, each one a divergence that somebody invented independently, each one a special case of the same equation.

```python
from divergence import cressie_read_divergence
cr = cressie_read_divergence(p, q, lambda_param=0.0)  # KL at lambda=0
```

---

## Rényi's telescope

Alfréd Rényi presented his generalization at the Fourth Berkeley Symposium in 1961. Where Shannon defined one entropy, Rényi defined a family indexed by a parameter $\alpha$:

$$H_\alpha(X) = \frac{1}{1-\alpha} \log \sum_x p(x)^\alpha$$

The parameter $\alpha$ acts like a focusing dial. Turn it one way and you count outcomes. Turn it the other and you zoom in on the peak.

- $\alpha \to 0$: Hartley entropy. Just counts how many outcomes are possible, ignoring their probabilities.
- $\alpha \to 1$: Shannon entropy. The familiar average surprise.
- $\alpha = 2$: Collision entropy. The probability that two independent draws give the same outcome.
- $\alpha \to \infty$: Min-entropy. Determined entirely by the single most likely event. Used in cryptography because it measures the worst case for an attacker.

Rényi entropy is monotonically non-increasing in $\alpha$. As you increase the parameter, you focus more on the high-probability events and the entropy drops. Every value of $\alpha$ gives you a different view of the same distribution.

[IMAGE: beyond_kl/renyi_entropy_alpha.png]

The four special cases are marked on the curve. You can watch Shannon entropy sitting between Hartley (the generous view) and min-entropy (the paranoid view).

```python
from divergence import renyi_entropy
h_shannon = renyi_entropy(samples, alpha=1.0)
h_min = renyi_entropy(samples, alpha=100.0)
```

---

## Mutual information: what correlation misses

Correlation measures linear association. If the relationship between X and Y is a straight line plus noise, correlation captures it well. If the relationship is nonlinear, correlation can return zero even when X completely determines Y.

Mutual information measures all statistical dependence. Linear, nonlinear, monotonic, non-monotonic. If knowing X tells you anything at all about Y, mutual information is positive.

The standard demonstration: generate Y = sin(2πX) + noise. Pearson correlation is near zero because the positive and negative half-cycles cancel. KSG mutual information (a kNN-based estimator developed by Kraskov, Stögbauer, and Grassberger in 2004) picks up the dependence immediately.

[IMAGE: distances_and_testing/correlation_vs_mutual_information.png]

The practical consequence: if you're selecting features for a model using correlation, you could be discarding exactly the variables that carry the most information.

```python
from divergence import ksg_mutual_information
mi = ksg_mutual_information(x, y, k=5)
```

---

## What you can do with all of this

The ideas above aren't just theory. Here's what they look like as tools:

**Detect causal direction.** Transfer entropy measures directed information flow between time series. Applied to stock market data, it can tell you which market leads and which follows.

**Test whether two samples come from the same distribution.** Formal two-sample permutation tests using energy distance or MMD, with calibrated p-values. No distributional assumptions required.

**Check MCMC convergence beyond R-hat.** Distributional chain comparison that catches failures in shape and tail behavior, not just means and variances.

**Measure what an experiment taught you.** Information gain (KL from prior to posterior) quantifies exactly how much the data updated your beliefs about each parameter.

**Decompose prediction uncertainty.** Separate the noise you can't reduce from the model uncertainty that more data would fix.

These are available in one Python package. 83 functions. Discrete and continuous. Works with ArviZ, PyMC, Stan, NumPyro, and emcee.

```bash
pip install divergence
```

Seven interactive notebooks walk through everything with real data, worked examples, and the stories of the people who built this mathematics:

github.com/michaelnowotny/divergence

---

Shannon wrote his paper in 1948. Seventy-seven years later, most working scientists still reach for Pearson's r when they want to measure association and chi-squared when they want to compare distributions. These are good tools. They're also the narrowest members of families that contain dozens of alternatives, each with different strengths.

The mathematics exists. It's been sitting in textbooks and journal papers for decades, waiting for someone to put it in one place. I think it's time more people had access to it.

*Built by Michael Nowotny. MIT license. Docs at michaelnowotny.github.io/divergence*
