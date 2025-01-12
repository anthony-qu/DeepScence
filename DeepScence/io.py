import os, numbers, math, anndata
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from DeepScence import logger
import pkg_resources
from scipy.stats import zscore, ks_2samp
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import curve_fit
from kneed import KneeLocator

import seaborn as sns


def read_dataset(adata, species, n=5, calculate_avg_exp=True, verbose=False):
    # check raw counts
    X_subset = adata.X[:10]
    if sp.sparse.issparse(X_subset):
        X_subset = X_subset.toarray()
    is_raw_counts = np.all(X_subset.astype(int) == X_subset)

    if verbose:
        if is_raw_counts:
            logger.info(
                f"Input is raw count, preprocessed {adata.n_vars} genes and {adata.n_obs} cells."
            )
        else:
            logger.info(
                f"Input is preprocessed, preprocessed {adata.n_vars} genes and {adata.n_obs} cells."
            )

    # get geneset

    geneset = get_geneset(n, species)

    # log normalize, subset, and scale
    out = adata.copy()
    if "log1p" in out.uns_keys():
        del out.uns["log1p"]
    out = normalize(out, geneset, verbose=verbose)

    return out


def normalize(adata, geneset, verbose):
    normalized = adata.copy()
    normalized.layers["raw_counts"] = normalized.X.copy()  # storing raw input

    # LorNormalize
    sc.pp.normalize_total(normalized, key_added="n_counts")
    normalized.obs["size_factors"] = normalized.obs.n_counts / np.median(
        normalized.obs.n_counts
    )

    # calculate CV
    raw_data = normalized.X
    gene_mean = np.mean(raw_data, axis=0)
    gene_std = np.std(raw_data, axis=0)
    normalized.var["cv"] = gene_std / gene_mean

    # log
    sc.pp.log1p(normalized)

    # subset
    genelist = geneset["gene_symbol"].values
    idx = np.where(normalized.var.index.isin(genelist))[0]
    assert (
        len(idx) > 5
    ), f"Too few genes ({len(idx)} genes) in the gene set are present in adata."
    normalized = normalized[:, idx]

    # calculate low variable genes among core genes
    normalized.var["low_variable"] = normalized.var["cv"] < np.percentile(
        normalized.var["cv"], 0
    )

    if verbose:
        logger.info("Using {} genes in the gene set for scoring".format(len(idx)))
    # scale
    sc.pp.scale(normalized)
    return normalized


# not using this
def get_avgexp(adata, direction_n):
    # given a lognormalized, subsetted and scaled adata.X
    file_path = pkg_resources.resource_filename("DeepScence", "data/coreGS_v2.csv")
    core = pd.read_csv(file_path, index_col=0)
    core = core[
        (core["gene_symbol"].isin(adata.var_names)) & (core["occurrence"] >= 5)
    ].dropna(subset=["direction"])
    # filter lowly variable genes
    core = core[~core["gene_symbol"].isin(adata.var_names[adata.var["low_variable"]])]

    # Subset the adata to include only the genes in the core data
    up_genes_5 = core[core["direction"] == "up"]["gene_symbol"].values
    down_genes_5 = core[core["direction"] == "down"]["gene_symbol"].values
    up_genes_6 = core[(core["occurrence"] >= 6) & (core["direction"] == "up")][
        "gene_symbol"
    ].values
    down_genes_6 = core[(core["occurrence"] >= 6) & (core["direction"] == "down")][
        "gene_symbol"
    ].values

    mean_up_5 = (
        adata[:, up_genes_5].X.mean(axis=1)
        if len(up_genes_5) > 0
        else np.full((adata.shape[0],), np.nan)
    )
    mean_down_5 = (
        adata[:, down_genes_5].X.mean(axis=1)
        if len(down_genes_5) > 0
        else np.full((adata.shape[0],), np.nan)
    )
    mean_up_6 = (
        adata[:, up_genes_6].X.mean(axis=1)
        if len(up_genes_6) > 0
        else np.full((adata.shape[0],), np.nan)
    )
    mean_down_6 = (
        adata[:, down_genes_6].X.mean(axis=1)
        if len(down_genes_6) > 0
        else np.full((adata.shape[0],), np.nan)
    )
    cdkn1a_exp = adata[:, "CDKN1A"].X.flatten()

    return mean_up_5, mean_down_5, mean_up_6, mean_down_6, cdkn1a_exp


def calculate_correlation(seneScore, adata, direction_n, species):
    correlation_results = []
    for i, gene in enumerate(adata.var_names):
        exp = adata.X[:, i]
        corr, _ = pearsonr(exp, seneScore)
        correlation_results.append({"gene_symbol": gene, "correlation": corr})
    correlation_results = pd.DataFrame(correlation_results)
    correlation_results["cv"] = adata.var["cv"].values
    correlation_results["low_variable"] = adata.var["low_variable"].values
    correlation_results = correlation_results.sort_values(
        by="correlation", ascending=False
    ).reset_index(drop=True)
    correlation_results = pd.merge(
        correlation_results,
        get_geneset(5, species)[["gene_symbol", "direction", "occurrence"]],
        on="gene_symbol",
        how="left",
    )
    correlation_results = correlation_results[
        (correlation_results["occurrence"] >= 5)
        & (correlation_results["low_variable"] == False)
    ]

    return correlation_results


def fix_score_direction(scores, adata, n, species):

    if "CDKN1A" in adata.var_names:
        anchor = "CDKN1A"
    elif "Cdkn1a" in adata.var_names:
        anchor = "Cdkn1a"
    else:
        raise ValueError("anchor missing.")
    corr_metrics = []
    reverse_log = []
    corr_dfs = []
    details = []

    for i in range(scores.shape[1]):
        seneScore = scores[:, i]
        corr_df = calculate_correlation(seneScore, adata, n, species)
        if corr_df.loc[corr_df["gene_symbol"] == "CDKN1A", "correlation"].values < 0:
            # if corr_df.index[corr_df["gene_symbol"] == anchor][0] > len(corr_df) / 2:
            reverse_log.append(True)
            seneScore = -seneScore
            corr_df = calculate_correlation(seneScore, adata, n, species)
        else:
            reverse_log.append(False)

        metric = corr_df["correlation"].abs().mean()
        # metric = corr_df[corr_df["correlation"] > 0.1]["correlation"].mean()
        corr_metrics.append(metric)
        corr_dfs.append(corr_df)
        scores[:, i] = seneScore

    if len(corr_metrics) == 2:
        node = np.argmax(corr_metrics)
        correlation, _ = pearsonr(scores[:, 0], scores[:, 1])
    else:
        node = 0
        correlation = "One node, no correlation"
    final_score = scores[:, node]

    log = {
        "corr_metrics": corr_metrics,
        "reverse_log": reverse_log,
        "node": node,
        "reverse": reverse_log[node],
        "correlation": correlation,
        "corr_df": corr_dfs[node],
    }

    cdkn1a_exp = adata[:, anchor].X.flatten()

    return final_score, log, cdkn1a_exp


def get_geneset(n=5, species="human"):
    file_path = pkg_resources.resource_filename("DeepScence", "data/coreGS_v2.csv")
    gs = pd.read_csv(file_path, index_col=0)
    gs = gs[gs["occurrence"] >= n]
    if species == "human":
        gs["gene_symbol"] = gs["gene_symbol_human"]
    elif species == "mouse":
        gs["gene_symbol"] = gs["gene_symbol_mouse"]
    else:
        raise ValueError("Species not supported. Please specify 'human' or 'mouse'.")
    return gs


def binarize_adata(adata, verbose=True):
    data = adata.obs["ds"].values.reshape(-1, 1)
    means_init = np.array([[np.mean(data)], [np.percentile(data, 90)]])
    gmm = GaussianMixture(
        n_components=2, covariance_type="full", random_state=0, means_init=means_init
    )
    gmm.fit(data)
    assignments = gmm.predict(data)
    max_assignment = assignments[np.argmax(adata.obs["ds"])]
    sorted_indices = adata.obs["ds"].argsort().values
    for idx in reversed(sorted_indices):
        if assignments[idx] != max_assignment:
            threshold = adata.obs["ds"].iloc[idx]
            break
    adata.obs["binary"] = np.where(adata.obs["ds"] > threshold, "SnC", "Normal")

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))  # One row, two columns
    if "SnC" not in adata.obs.columns:
        adata.obs["SnC"] = "in-vivo"
    colors = ["#ff1b6b", "#45caff"]

    # Plot 1: ds distribution for in-vitro
    sns.histplot(
        adata.obs["ds"],
        kde=False,
        ax=axes[0],
        color="lightgrey",
        bins=50,
        stat="density",
    )
    for idx, level in enumerate(adata.obs["SnC"].unique()):
        sns.kdeplot(
            adata.obs["ds"][adata.obs["SnC"] == level],
            ax=axes[0],
            label=level,
            color=colors[idx],
        )
    axes[0].set_title("Score distribution by condition")
    axes[0].legend(title="SnC Level")
    axes[0].set_xlabel("ds")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(adata.obs["ds"].min(), adata.obs["ds"].max())

    # Plot 2: binarization results
    x = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    pdf1 = np.exp(gmm.score_samples(x))
    axes[1].hist(data, bins=30, density=True, alpha=0.5, color="gray", label="Data")
    axes[1].plot(x, pdf1, label="Mixture of 2 Gaussians", color="red")

    weights = gmm.weights_
    means = gmm.means_.flatten()
    covariances = np.sqrt(gmm.covariances_).flatten()

    for weight, mean, cov in zip(weights, means, covariances):
        pdf = (
            weight
            * (1 / (np.sqrt(2 * np.pi) * cov))
            * np.exp(-0.5 * ((x - mean) / cov) ** 2)
        )
        axes[1].plot(x, pdf, label=f"Gaussian: μ={mean:.2f}, σ={cov:.2f}")
    axes[1].axvline(x=threshold, color="blue", linestyle="--")
    axes[1].set_xlabel("DeepScence Scores")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Mixture")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return adata
