import anndata
import numpy as np
import scanpy as sc
import scipy as sp
import torch
from .io import (
    read_dataset,
    normalize,
    fix_score_direction,
    binarize_adata,
    permute,
    plot_heatmaps,
)
from .train import train
from .network import ZINBAutoencoder, Autoencoder
import random
import pandas as pd
import os
import pkg_resources
import warnings
from DeepScence import logger
from dca.api import dca
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def DeepScence(
    adata,
    lambda_ortho=0.1,
    hidden_sizes=(32,),
    batchnorm=False,
    dropout=0.5,
    epochs=300,
    validation_split=0.1,
    reduce_lr=10,
    early_stop=25,
    batch_size=None,
    learning_rate=0.005,
    n=5,
    direction_n=5,
    random_state=0,
    verbose=False,
    binarize=False,
    denoise=True,
):

    assert isinstance(adata, anndata.AnnData), "adata must be an AnnData instance"

    # set seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = "0"
    torch.manual_seed(random_state)

    sc.pp.filter_genes(adata, min_cells=1)

    # coerce adata.X to dense
    if sp.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()

    # save the original version
    original = adata.copy()

    # denoise
    if denoise:
        dca(adata, random_state=random_state)

    # read adata, subset, calculate up/down metrics
    adata = read_dataset(adata, n, direction_n, verbose=True)

    input_size = adata.n_vars
    model = ZINBAutoencoder(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        size_factors=adata.obs["size_factors"],
        batchnorm=batchnorm,
        dropout=dropout,
        lambda_ortho=lambda_ortho,
    )

    if lambda_ortho is not None:
        logger.info("Lambda provided, capturing scores in 2 neurons.")

    train(
        model,
        adata,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        reduce_lr=reduce_lr,
        early_stop=early_stop,
        verbose=verbose,
    )

    # scores = model.encoded_scores.detach().cpu().numpy()
    scores = model.predict(adata)

    scores, log, cdkn1a_exp = fix_score_direction(scores, adata, direction_n)
    original.obsm["CDKN1A"] = cdkn1a_exp
    original.obs["ds"] = scores
    original.uns["log"] = log

    # add binarization by directly calculating scores from trained model
    if binarize:
        # plot_heatmaps(read_dataset(original))
        n_perm = 50
        scores_perm_all = []

        logger.info("Binarizing with permutation...")
        for i in tqdm(range(n_perm)):
            np.random.seed(random_state + i)
            adata_perm = permute(
                original,
                n=n,
                sene_genes_only=True,
                target_sum=None,
                permute_together=True,
            )

            adata_perm = read_dataset(
                adata_perm, n=n, direction_n=direction_n, calculate_avg_exp=False
            )
            scores_perm = model.predict(adata_perm)[:, log["node"]]
            if log["reverse"]:
                scores_perm = -scores_perm
            scores_perm_all.append(scores_perm)

        scores_perm_mean = np.mean(np.array(scores_perm_all), axis=0)
        original.obsm["scores_perm_mean"] = scores_perm_mean

        # use perm scores to estimate
        original = binarize_adata(
            original, scores_perm_all, mean_level=True, verbose=verbose
        )

    original.X = sp.sparse.csr_matrix(original.X)
    return original
