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


def read_dataset(adata, n=5, calculate_avg_exp=True, verbose=False):
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

    geneset = get_geneset(n)

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


def calculate_correlation(seneScore, adata, direction_n):
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
        get_geneset(5)[["gene_symbol", "direction", "occurrence"]],
        on="gene_symbol",
        how="left",
    )
    correlation_results = correlation_results[
        (correlation_results["occurrence"] >= 5)
        & (correlation_results["low_variable"] == False)
    ]

    return correlation_results


def fix_score_direction(scores, adata, n):

    corr_metrics = []
    reverse_log = []
    corr_dfs = []
    details = []

    for i in range(scores.shape[1]):
        seneScore = scores[:, i]
        corr_df = calculate_correlation(seneScore, adata, n)
        # if corr_df.loc[corr_df["gene_symbol"] == "CDKN1A", "correlation"].values < 0:
        if corr_df.index[corr_df["gene_symbol"] == "CDKN1A"][0] > len(corr_df) / 2:
            reverse_log.append(True)
            seneScore = -seneScore
            corr_df = calculate_correlation(seneScore, adata, n)
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

    cdkn1a_exp = adata[:, "CDKN1A"].X.flatten()

    return final_score, log, cdkn1a_exp


def get_geneset(n=5):
    file_path = pkg_resources.resource_filename("DeepScence", "data/coreGS_v2.csv")
    gs = pd.read_csv(file_path, index_col=0)
    gs = gs[gs["occurrence"] >= n]
    return gs


def permute(
    adata,
    n=5,
    sene_genes_only=True,
    target_sum=None,
    permute_together=False,
):
    data = adata.X.copy()
    geneset = get_geneset(n=n)
    geneset_indices = adata.var.index.isin(geneset["gene_symbol"])
    gene_sums = np.sum(data, axis=0)

    if target_sum is None:
        target_sum = np.median(gene_sums)
    size_factors = gene_sums / target_sum
    data = data / size_factors

    if permute_together:
        # Method 2: Reorder the rows (genes) for all cells uniformly
        if sene_genes_only:
            subset = data[:, geneset_indices]
            np.random.shuffle(subset)
            data[:, geneset_indices] = subset
        else:
            np.random.shuffle(data)
    else:
        # Method 1: For each cell, permute its values across genes
        for i in range(data.shape[0]):
            if sene_genes_only:
                subset = data[i, geneset_indices]
                np.random.shuffle(subset)
                data[i, geneset_indices] = subset
            else:
                np.random.shuffle(data[i, :])

    data = data * size_factors.flatten()  # denormalize

    # Create a new AnnData object with the permuted data
    permuted_adata = anndata.AnnData(X=data, var=adata.var.copy(), obs=adata.obs.copy())
    permuted_adata.obs_names = pd.Index([f"{name}_perm" for name in adata.obs_names])
    return permuted_adata


def get_elbow(xdata, ydata):
    # fit sigmoid curve, get x axis of elbow.
    def sigmoid(x, x0, k):
        return 1 / (1 + np.exp(-k * (x - x0)))

    def sigmoid_prime(x, x0, k):
        exp_term = np.exp(-k * (x - x0))
        return k * exp_term / (1 + exp_term) ** 2

    def sigmoid_double_prime(x, x0, k):
        exp_term = np.exp(-k * (x - x0))
        return k**2 * exp_term * (1 - exp_term) / (1 + exp_term) ** 3

    def get_curvature(x, x0, k):
        f_prime = sigmoid_prime(x, x0, k)
        f_double_prime = sigmoid_double_prime(x, x0, k)
        curvature_val = np.abs(f_double_prime) / (1 + f_prime**2) ** (3 / 2)

        return curvature_val

    p0 = [np.median(xdata), -2.0]
    bounds = ([min(xdata), -np.inf], [max(xdata), 0])
    popt, pcov = curve_fit(sigmoid, xdata, ydata, bounds=bounds)
    x_0, k = popt
    x_fit = np.linspace(min(xdata), max(xdata), 500)

    y_fit = sigmoid(x_fit, *popt)
    curvature = get_curvature(x_fit, *popt)
    d2 = sigmoid_double_prime(x_fit, *popt)

    # # curvature max
    # y_half_idx = np.abs(y_fit - 0.5).argmin()  # find midpoint
    # x_fit_after = x_fit[y_half_idx:]  # look at only right half
    # curvature_after = curvature[y_half_idx:]
    # elbow2 = x_fit_after[np.argmax(np.nan_to_num(curvature_after, nan=-np.inf))]

    # 2nd derivative max
    elbow2 = x_fit[np.argmin(d2)]
    return x_fit, y_fit, elbow2, popt


def get_snc_clusters(adata, threshold):
    df = adata.obs[["ds"]]
    n_components = np.arange(2, 5 + 1)  # Test from 2 to 10 components
    models = []
    aics = []

    # Iterate over the range of n_components and fit the models
    for n in n_components:
        try:
            model = GaussianMixture(n, covariance_type="full", random_state=0).fit(df)
            models.append(model)
            aics.append(model.aic(df))  # Store AIC if fit is successful
        except Exception as e:
            print(f"Error with n_components = {n}")
            models.append(None)  # Store None for this model
            aics.append(np.nan)  # Append NaN for AIC to mark the failure

    # plt.plot(n_components, aics, marker="o")
    # plt.title("AIC Scores by Number of Components")
    # plt.xlabel("Number of components")
    # plt.ylabel("AIC")
    # plt.show()
    best_model = models[np.nanargmin(aics)]

    # best_model = BayesianGaussianMixture(
    #     n_components=10, covariance_type="tied", random_state=0
    # ).fit(df)

    # find target clusters
    adata.obs["cluster"] = best_model.predict(df)
    cluster_means = best_model.means_.flatten()
    cluster_stds = np.sqrt(best_model.covariances_.flatten())

    target_clusters = np.where(cluster_means - 1.645 * cluster_stds > threshold)[0]
    rightmost_cluster = np.argmax(cluster_means)
    # if len(target_clusters) == 0 and np.max(cluster_means) > threshold:
    #     target_clusters = np.array([rightmost_cluster])

    # check if the cell with the highest score is in the target cluster (edge case)
    highest_score_cluster = adata.obs.iloc[np.argmax(df["ds"])]["cluster"]
    if highest_score_cluster in target_clusters or len(target_clusters) == 0:
        if len(target_clusters) > 0:  # target contain the rightmost (normal case)
            min_score_in_target = adata.obs.loc[
                adata.obs["cluster"].isin(target_clusters), "ds"
            ].min()
        else:
            # no peak significant
            if np.max(cluster_means) > threshold:
                min_score_in_target = adata.obs.loc[
                    adata.obs["cluster"] == highest_score_cluster, "ds"
                ].min()
            else:
                min_score_in_target = float("inf")
        adata.obs["binary"] = (adata.obs["ds"] > min_score_in_target).astype(int)
    else:
        # Edge case: The highest scoring cell does not belong to the target peak
        sorted_cells = adata.obs.sort_values("ds", ascending=False).copy()
        sorted_cells["binary"] = 0

        for i in range(1, len(sorted_cells)):
            current_cluster = sorted_cells.iloc[i]["cluster"]
            previous_cluster = sorted_cells.iloc[i - 1]["cluster"]
            if (
                current_cluster != previous_cluster
                and current_cluster not in target_clusters
            ):
                sorted_cells.iloc[:i, sorted_cells.columns.get_loc("binary")] = 1
                break
        adata.obs["binary"] = sorted_cells["binary"].reindex(
            adata.obs.index, fill_value=0
        )
    return best_model, target_clusters
    # while True:
    #     best_model = GaussianMixture(
    #         n_component, covariance_type="diag", random_state=0
    #     ).fit(df)
    #     adata.obs["cluster"] = best_model.predict(df)

    #     cluster_means = best_model.means_.flatten()
    #     cluster_stds = np.sqrt(best_model.covariances_.flatten())
    #     target_clusters = np.where(cluster_means - 1.96 * cluster_stds > threshold)[0]
    #     highest_mean_cluster = np.argmax(cluster_means)

    #     if len(target_clusters) > 0:
    #         if highest_mean_cluster not in target_clusters:
    #             target_clusters = np.array([highest_mean_cluster])
    #         print(f"Final n component: {n_component}")
    #         break
    #     else:
    #         print("shit")
    #         n_component += 1
    #         if n_component > 10:
    #             print("Maximum of 10 components reached. No SnC found.")
    #             target_clusters = []
    #             break


def binarize_adata(adata, scores_perm_all, mean_level=False, verbose=True):
    scores_perm_all = np.array(scores_perm_all)
    # binarize in cell level

    if mean_level:
        perm_scores = adata.obsm["scores_perm_mean"]
    else:
        perm_scores = np.array(scores_perm_all).flatten()
    mean_perm = perm_scores.mean()
    std_perm = perm_scores.std()
    # ds_threshold = mean_perm + 1.645 * std_perm

    # CI based method
    ps = []
    for i in range(scores_perm_all.shape[1]):
        p = np.mean(scores_perm_all[:, i] > adata.obs["ds"].iloc[i])
        ps.append(p)
    adata.obs["p"] = ps

    mask = (adata.obs["p"] > 0.8) | (adata.obs["p"] < 0.2)
    xdata = adata.obs["ds"].values[mask]
    ydata = adata.obs["p"].values[mask]
    x_fit, y_fit, elbow2, popt = get_elbow(xdata, ydata)
    # ds_threshold = elbow2
    ds_threshold = x_fit[(abs(y_fit - 0.5)).argmin()]

    # moving elbow to the right
    sorted_ds = np.sort(adata.obs["ds"].unique())
    for ds_threshold in sorted_ds[sorted_ds >= elbow2]:
        num_bad_sncs = (adata.obs["p"][adata.obs["ds"] >= ds_threshold] >= 0.5).sum()
        pt_bad_sncs = np.mean(adata.obs["p"][adata.obs["ds"] >= ds_threshold] >= 0.5)
        if pt_bad_sncs < 0.02:
            break
    # adata.obs["binary"] = (adata.obs["ds"] >= ds_threshold).astype(int)

    # clustering
    best_model, target_clusters = get_snc_clusters(adata, ds_threshold)
    prop = np.mean(adata.obs["binary"] == 1)

    if verbose:
        # plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 7))
        if "SnC" not in adata.obs.columns:
            adata.obs["SnC"] = "in-vivo"
        colors = ["#ff1b6b", "#45caff"]
        # plot 1: ds distribution for in-vitro.
        sns.histplot(
            adata.obs["ds"],
            kde=False,
            ax=axes[0, 0],
            color="lightgrey",
            bins=50,
            stat="density",
        )
        for idx, level in enumerate(adata.obs["SnC"].unique()):
            sns.kdeplot(
                adata.obs["ds"][adata.obs["SnC"] == level],
                ax=axes[0, 0],
                label=level,
                color=colors[idx],
            )
        axes[0, 0].set_title("Score distribution by condition")
        axes[0, 0].legend(title="SnC Level")
        axes[0, 0].set_xlabel("ds")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_xlim(adata.obs["ds"].min(), adata.obs["ds"].max())

        # plot 2: peaks
        palette = {0: "#45caff", 1: "#ff1b6b"}

        # peaks
        x = np.linspace(adata.obs["ds"].min(), adata.obs["ds"].max(), 1000).reshape(
            -1, 1
        )
        log_prob = best_model.score_samples(x)
        responsibilities = best_model.predict_proba(x)
        pdf = np.exp(log_prob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

        sns.histplot(
            adata.obs["ds"],
            kde=False,
            ax=axes[0, 1],
            color="lightgrey",
            bins=30,
            stat="density",
        )
        axes[0, 1].axvline(x=ds_threshold, color="red", linestyle="-")
        for i in range(best_model.n_components):
            color = "lightgrey" if i not in target_clusters else "blue"
            sns.lineplot(
                x=x.squeeze(), y=pdf_individual[:, i], ax=axes[0, 1], color=color
            )
        means = best_model.means_.squeeze()
        y_max = axes[0, 1].get_ylim()[1]

        for i, mean in enumerate(means):
            color = "blue" if i in target_clusters else "lightgrey"
            axes[0, 1].axvline(mean, linestyle="--", color=color)
            axes[0, 1].text(
                mean,
                y_max * 0.9,
                f"{mean:.2f}",
                color="red" if i in target_clusters else "lightgrey",
                ha="center",
                va="center",
            )

        axes[0, 1].set_title("GMM best fit for scores")
        axes[0, 1].set_xlabel("ds")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].set_ylim(0, y_max)
        axes[0, 1].set_xlim(adata.obs["ds"].min(), adata.obs["ds"].max())

        # plot 3: score vs. p-val
        sns.scatterplot(
            data=adata.obs,
            x="ds",
            y="p",
            hue="binary",
            palette=palette,
            s=10,
            legend="full",
            ax=axes[1, 0],
        )
        axes[1, 0].plot(x_fit, y_fit, color="red", label="Reverse Sigmoid Fit")
        axes[1, 0].set_xlim(adata.obs["ds"].min(), adata.obs["ds"].max())
        axes[1, 0].axvline(
            x=ds_threshold,
            color="blue",
            linestyle="--",
            label=f"x = {ds_threshold:.2f}",
        )
        axes[1, 0].text(
            ds_threshold,
            0.5,
            f"ds_threshold: {ds_threshold:.2f}",
        )

        # sns.histplot(perm_scores, bins=50, kde=False, ax=axes[1, 0], color="#9BA1F3")
        # axes[1, 0].axvline(
        #     x=ds_threshold, color="red", linestyle="--", label=f"x = {ds_threshold}"
        # )
        # axes[1, 0].set_xlim(adata.obs["ds"].min(), adata.obs["ds"].max())
        # axes[1, 0].set_title("scores_perm_all")
        # axes[1, 0].set_xlabel(None)
        # axes[1, 0].set_ylabel(None)

        # plot 4: score vs. CKDN1A
        sns.scatterplot(
            data=adata.obs,
            x=adata.obs["ds"],
            y=adata.obsm["CDKN1A"],
            hue=adata.obs["binary"],
            palette=palette,
            s=10,  # Adjust size of points
            legend="full",
            ax=axes[1, 1],
        )
        axes[1, 1].set_title(f"Determined SnCs ({100*prop:.3f}%)")
        axes[1, 1].set_xlabel("ds")
        axes[1, 1].set_ylabel("CDKN1A expression")
        axes[1, 1].set_xlim(adata.obs["ds"].min(), adata.obs["ds"].max())

        plt.tight_layout()
        plt.show()

    return adata


def plot_heatmaps(input_adata):
    # take in a lognormalized scaled adata
    geneset = get_geneset(5)
    geneset = geneset[geneset["gene_symbol"].isin(input_adata.var_names)].sort_values(
        by="direction"
    )
    genes = geneset["gene_symbol"].values

    if input_adata.n_obs > 10000:
        input_adata = input_adata[
            np.random.choice(input_adata.n_obs, 10000, replace=False), :
        ]

    if "SnC" in input_adata.obs.columns:
        normal = input_adata[input_adata.obs["SnC"] == 0].copy()
        snc = input_adata[input_adata.obs["SnC"] == 1].copy()
    else:
        sorted_adata = input_adata[input_adata.obs["ds"].argsort()].copy()
        split_point = input_adata.n_obs // 2
        normal = sorted_adata[:split_point].copy()
        snc = sorted_adata[split_point:].copy()

    # sort by ds
    normal = normal[np.argsort(normal.obs["ds"].values), :]
    snc = snc[np.argsort(snc.obs["ds"].values), :]

    # Extract expression data for geneset genes
    normal_data = normal[:, genes].X.toarray().T
    snc_data = snc[:, genes].X.toarray().T

    # Extract ds values for x-axis labels
    normal_ds = np.round(normal.obs["ds"].values, 2)
    snc_ds = np.round(snc.obs["ds"].values, 2)

    # Plot heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    vmin = min(normal_data.min(), snc_data.min())
    # vmax = max(normal_data.max(), snc_data.max())
    vmax = np.percentile(
        np.concatenate([normal_data.flatten(), snc_data.flatten()]), 90
    )

    sns.heatmap(
        normal_data,
        ax=axes[0],
        cmap="viridis",
        cbar=False,
        yticklabels=genes,
        xticklabels=normal_ds,
        vmin=vmin,
        vmax=vmax,
    )
    step = len(normal_ds) // 20
    axes[0].set_title("Original normal")
    axes[0].set_xticks(np.arange(0, len(normal_ds), step))
    axes[0].set_xticklabels(normal_ds[::step], rotation=0)

    sns.heatmap(
        snc_data,
        ax=axes[1],
        cmap="viridis",
        cbar=False,
        yticklabels=False,
        xticklabels=snc_ds,
        vmin=vmin,
        vmax=vmax,
    )
    step = len(snc_ds) // 20
    axes[1].set_title("Original SnC")
    axes[1].set_xticks(np.arange(0, len(snc_ds), step))
    axes[1].set_xticklabels(snc_ds[::step], rotation=0)

    plt.tight_layout()
    plt.show()
