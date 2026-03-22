"""Generate high-resolution figures for X Articles.

Run from the repo root:
    python scripts/generate_article_figures.py

Produces 300 DPI PNGs in notebooks/figures/article_ready/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

# Style for article figures: larger text, thicker lines, clean background
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

OUT = Path("notebooks/figures/article_ready")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)


# =========================================================================
# ARTICLE 1 FIGURES
# =========================================================================

# --- 1. Crop yield heat cliff ---
print("1. Crop yield heat cliff...")
n = 500
temp = rng.uniform(15, 38, n)
yield_base = np.where(temp < 29, 50 + 8 * (temp - 15), 162 - 15 * (temp - 29))
yield_obs = yield_base + rng.normal(0, 15, n)

from divergence import ksg_mutual_information

r_val, _ = pearsonr(temp, yield_obs)
mi_val = ksg_mutual_information(temp, yield_obs, k=5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

ax1.scatter(temp, yield_obs, alpha=0.4, s=20, color="#2ecc71", edgecolors="none")
t_sorted = np.linspace(15, 38, 200)
y_true = np.where(t_sorted < 29, 50 + 8 * (t_sorted - 15), 162 - 15 * (t_sorted - 29))
ax1.plot(t_sorted, y_true, "k-", lw=3, label="True response")
ax1.axvline(29, color="#e74c3c", ls="--", lw=2, alpha=0.7, label="Heat threshold (29°C)")
ax1.set_xlabel("Growing season temperature (°C)")
ax1.set_ylabel("Corn yield (bushels/acre)")
ax1.set_title("The heat cliff: yields collapse above 29°C")
ax1.legend(loc="upper left")

ax2.bar(
    ["Pearson |r|", "Mutual\ninformation\n(nats)"],
    [abs(r_val), mi_val],
    color=["#e74c3c", "#2ecc71"],
    edgecolor="white",
    linewidth=2,
    width=0.55,
)
ax2.set_title("Correlation sees nothing.\nMutual information sees the cliff.")
for i, v in enumerate([abs(r_val), mi_val]):
    ax2.text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=18, fontweight="bold")

plt.tight_layout(w_pad=3)
fig.savefig(OUT / "crop_yield_heat_cliff.png")
plt.close()


# --- 2. Stock market contagion ---
print("2. Stock market contagion...")
import yfinance as yf
from divergence import transfer_entropy

sp500 = yf.download("^GSPC", start="2020-01-01", end="2024-01-01", progress=False)
nikkei = yf.download("^N225", start="2020-01-01", end="2024-01-01", progress=False)

sp_ret = np.log(sp500["Close"]).diff().dropna()
nk_ret = np.log(nikkei["Close"]).diff().dropna()
common = sp_ret.index.intersection(nk_ret.index)
sp = sp_ret.loc[common].values.ravel()
nk = nk_ret.loc[common].values.ravel()

r_stock, _ = pearsonr(sp, nk)
te_sp_nk = transfer_entropy(source=sp, target=nk, k=1, lag=1)
te_nk_sp = transfer_entropy(source=nk, target=sp, k=1, lag=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

ax1.scatter(sp * 100, nk * 100, alpha=0.25, s=12, color="#3498db", edgecolors="none")
ax1.set_xlabel("S&P 500 daily return (%)")
ax1.set_ylabel("Nikkei 225 daily return (%)")
ax1.set_title(f"Correlation: r = {r_stock:.3f}\n(symmetric — no direction)")
ax1.axhline(0, color="gray", lw=0.5)
ax1.axvline(0, color="gray", lw=0.5)

bars = ax2.bar(
    ["S&P → Nikkei", "Nikkei → S&P"],
    [te_sp_nk, max(te_nk_sp, 0)],
    color=["#3498db", "#e74c3c"],
    edgecolor="white",
    linewidth=2,
    width=0.5,
)
for bar, val in zip(bars, [te_sp_nk, te_nk_sp]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.4f}",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )
ax2.set_ylabel("Transfer entropy (nats)")
ax2.set_title("Transfer entropy:\ndirected information flow")

plt.tight_layout(w_pad=3)
fig.savefig(OUT / "stock_market_contagion.png")
plt.close()


# --- 3. Information gain (Nile) ---
print("3. Information gain (Nile)...")
# Recreate from the Bayesian notebook data
ig_names = ["$\\mu_1$\n(pre-change)", "$\\mu_2$\n(post-change)", "$\\log\\sigma$\n(variability)", "$\\tau$\n(change year)"]
ig_vals = [2.49, 2.99, 2.12, 3.69]  # from the notebook output
colors_ig = ["#3498db", "#f39c12", "#2ecc71", "#e74c3c"]

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.bar(ig_names, ig_vals, color=colors_ig, edgecolor="white", linewidth=2, width=0.6)
for i, v in enumerate(ig_vals):
    ax.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=18, fontweight="bold")
ax.set_ylabel("Information gain (nats)")
ax.set_title("How much did the Nile data teach us\nabout each parameter?")
ax.set_ylim(0, max(ig_vals) * 1.2)

fig.savefig(OUT / "information_gain.png")
plt.close()


# --- 4. R-hat failure ---
print("4. R-hat failure...")
import arviz as az
from divergence import chain_two_sample_test

rng_rhat = np.random.default_rng(42)
n_draws = 2000


def good_chain():
    return rng_rhat.permutation(
        np.concatenate(
            [rng_rhat.normal(0, 1, int(n_draws * 0.85)), rng_rhat.normal(3, 0.6, int(n_draws * 0.15))]
        )
    )


def bad_chain():
    return rng_rhat.normal(0.1, 1.05, n_draws)


chains_rhat = np.array([good_chain(), good_chain(), bad_chain(), bad_chain()])
idata_rhat = az.from_dict({"posterior": {"theta": chains_rhat}})
rhat_val = float(az.rhat(idata_rhat)["theta"].values)
ess_val = float(az.ess(idata_rhat)["theta"].values)

result_rhat = chain_two_sample_test(
    idata_rhat, var_names=["theta"], method="energy", n_permutations=200, seed=42
)
p_min = result_rhat["theta"].min_p_value

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

for c in range(4):
    color = "#3498db" if c < 2 else "#e74c3c"
    label = f'Chain {c} ({"correct" if c < 2 else "incomplete"})'
    ax1.hist(chains_rhat[c], bins=60, alpha=0.4, color=color, density=True, label=label)

ax1.axvline(3, color="black", ls="--", lw=2, alpha=0.5, label="Secondary mode")
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel("Density")
ax1.set_title(f"$\\hat{{R}}$ = {rhat_val:.3f}, ESS = {ess_val:.0f}\nStandard diagnostics: CONVERGED")
ax1.legend(fontsize=11)

im = ax2.imshow(
    result_rhat["theta"].p_value_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal"
)
plt.colorbar(im, ax=ax2, shrink=0.8, label="p-value")
ax2.set_xlabel("Chain")
ax2.set_ylabel("Chain")
ax2.set_title(f"Two-sample test p-values\nmin p = {p_min:.4f} — NOT CONVERGED")
for i in range(4):
    for j in range(4):
        p = result_rhat["theta"].p_value_matrix[i, j]
        color = "white" if p < 0.3 else "black"
        ax2.text(j, i, f"{p:.2f}", ha="center", va="center", fontsize=14, fontweight="bold", color=color)

plt.tight_layout(w_pad=3)
fig.savefig(OUT / "rhat_failure.png")
plt.close()


# =========================================================================
# ARTICLE 2 FIGURES
# =========================================================================

# --- 5. Two distributions ---
print("5. Two distributions...")
from scipy.stats import norm

x_grid = np.linspace(-4, 8, 500)
pdf_p = norm.pdf(x_grid, 2, 1.5)
pdf_q = norm.pdf(x_grid, 0, 1.0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_grid, pdf_p, lw=3, color="#3498db", label=r"$P = \mathcal{N}(2,\; 1.5^2)$")
ax.plot(x_grid, pdf_q, lw=3, color="#e74c3c", label=r"$Q = \mathcal{N}(0,\; 1.0^2)$")
ax.fill_between(x_grid, pdf_p, alpha=0.15, color="#3498db")
ax.fill_between(x_grid, pdf_q, alpha=0.15, color="#e74c3c")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("Two distributions: different means, different spreads")
ax.legend()

fig.savefig(OUT / "two_distributions.png")
plt.close()


# --- 6. Generator functions ---
print("6. f-divergence generator functions...")
t = np.linspace(0.05, 4, 300)

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.plot(t, t * np.log(t), lw=3, label="KL: $t \\log t$", color="#3498db")
ax.plot(t, (t - 1) ** 2, lw=3, label="$\\chi^2$: $(t-1)^2$", color="#e74c3c")
ax.plot(t, (np.sqrt(t) - 1) ** 2, lw=3, label="Hellinger: $(\\sqrt{t}-1)^2$", color="#2ecc71")
ax.plot(t, 0.5 * np.abs(t - 1), lw=3, label="TV: $|t-1|/2$", color="#f39c12")
ax.plot(t, -np.log(t), lw=3, label="Reverse KL: $-\\log t$", color="#9b59b6")
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(1, color="gray", lw=0.5, ls="--")
ax.set_xlabel("Density ratio $t = p/q$")
ax.set_ylabel("$f(t)$")
ax.set_title("Generator functions of the f-divergence family")
ax.legend()
ax.set_xlim(0, 4)
ax.set_ylim(-1.5, 5)

fig.savefig(OUT / "generator_functions.png")
plt.close()


# --- 7. Cressie-Read continuum ---
print("7. Cressie-Read continuum...")
from divergence import cressie_read_divergence

np.random.seed(42)
sample_p = np.concatenate([rng.normal(2, 1.5, 8000), -rng.normal(2, 1.5, 8000) + 4])
sample_q = np.concatenate([rng.normal(0, 1.0, 8000), -rng.normal(0, 1.0, 8000)])

lambdas = np.linspace(-1.4, 3.0, 200)
cr_vals = []
for lam in lambdas:
    try:
        val = cressie_read_divergence(sample_p, sample_q, lambda_param=lam)
        cr_vals.append(val if np.isfinite(val) and val < 20 else np.nan)
    except Exception:
        cr_vals.append(np.nan)

fig, ax = plt.subplots(figsize=(12, 6.5))
ax.plot(lambdas, cr_vals, lw=3, color="#3498db")

special = [
    (-1, "Reverse KL", "#9b59b6"),
    (-0.5, "Hellinger", "#2ecc71"),
    (0, "KL", "#e74c3c"),
    (1, "Pearson $\\chi^2$", "#f39c12"),
]
for lam_s, name, color in special:
    idx = np.argmin(np.abs(lambdas - lam_s))
    if not np.isnan(cr_vals[idx]):
        ax.plot(lam_s, cr_vals[idx], "o", markersize=12, color=color, zorder=5)
        ax.annotate(
            name,
            xy=(lam_s, cr_vals[idx]),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

ax.set_xlabel("$\\lambda$")
ax.set_ylabel("Cressie-Read divergence")
ax.set_title("One parameter connects KL, $\\chi^2$, Hellinger, and reverse KL")

fig.savefig(OUT / "cressie_read_continuum.png")
plt.close()


# --- 8. Rényi entropy ---
print("8. Rényi entropy telescope...")
from divergence import renyi_entropy, discrete_entropy

disc_sample = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4])
alphas = np.concatenate([np.linspace(0.01, 0.99, 50), np.linspace(1.01, 10, 100)])
renyi_vals = [renyi_entropy(disc_sample, alpha=a, discrete=True) for a in alphas]

h_shannon = discrete_entropy(disc_sample)

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.plot(alphas, renyi_vals, lw=3, color="#3498db")
ax.axhline(h_shannon, color="#e74c3c", ls="--", lw=2, alpha=0.7)

markers = [
    (0.01, renyi_vals[0], "$\\alpha \\to 0$: Hartley\n(counts outcomes)", "#2ecc71"),
    (1.0, h_shannon, "$\\alpha \\to 1$: Shannon", "#e74c3c"),
    (2.0, renyi_entropy(disc_sample, alpha=2.0, discrete=True), "$\\alpha = 2$: Collision", "#f39c12"),
    (10.0, renyi_vals[-1], "$\\alpha \\to \\infty$: Min-entropy", "#9b59b6"),
]
for a, h, label, color in markers:
    ax.plot(a, h, "o", markersize=10, color=color, zorder=5)
    offset = (12, 12) if a < 5 else (12, -20)
    ax.annotate(label, xy=(a, h), xytext=offset, textcoords="offset points",
                fontsize=12, color=color, fontweight="bold")

ax.set_xlabel("$\\alpha$")
ax.set_ylabel("Rényi entropy (nats)")
ax.set_title("Rényi entropy: one dial, four special cases")
ax.set_xlim(-0.2, 10.5)

fig.savefig(OUT / "renyi_entropy_telescope.png")
plt.close()


# --- 9. Correlation vs MI ---
print("9. Correlation vs mutual information...")
from divergence import ksg_mutual_information as ksg_mi

rng2 = np.random.default_rng(42)
n_mi = 1000
x_mi = rng2.uniform(0, 1, n_mi)
y_mi = np.sin(2 * np.pi * x_mi) + 0.3 * rng2.standard_normal(n_mi)

r_mi, _ = pearsonr(x_mi, y_mi)
mi_mi = ksg_mi(x_mi, y_mi, k=5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

ax1.scatter(x_mi, y_mi, alpha=0.4, s=15, color="#3498db", edgecolors="none")
x_curve = np.linspace(0, 1, 200)
ax1.plot(x_curve, np.sin(2 * np.pi * x_curve), "k-", lw=3, alpha=0.5, label="$y = \\sin(2\\pi x)$")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("$Y = \\sin(2\\pi X) + \\mathrm{noise}$")
ax1.legend()

ax2.bar(
    ["Pearson |r|", "KSG mutual\ninformation\n(nats)"],
    [abs(r_mi), mi_mi],
    color=["#e74c3c", "#3498db"],
    edgecolor="white",
    linewidth=2,
    width=0.55,
)
for i, v in enumerate([abs(r_mi), mi_mi]):
    ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=18, fontweight="bold")
ax2.set_title("Correlation is blind to\nnonlinear dependence")

plt.tight_layout(w_pad=3)
fig.savefig(OUT / "correlation_vs_mi.png")
plt.close()


# --- Summary ---
print()
print("Done! Article-ready figures in notebooks/figures/article_ready/:")
for f in sorted(OUT.glob("*.png")):
    from PIL import Image
    img = Image.open(f)
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name}: {img.width}x{img.height}, {size_kb:.0f} KB")
