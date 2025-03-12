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
)
from .train import train
from .network import ZINBAutoencoder, Autoencoder
import random
import pandas as pd
import os
import pkg_resources
import warnings
from DeepScence import logger
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def DeepScence(
    adata,
    binarize=False,
    species="human",
    custome_gs=None,
    anchor_gene=None,
    lambda_ortho=0.1,
    lambda_mmd=0.7,
    hidden_sizes=(32,),
    batchnorm=False,
    dropout=None,
    epochs=300,
    validation_split=0.1,
    reduce_lr=10,
    early_stop=35,
    batch_size=None,
    learning_rate=0.005,
    n=5,
    random_state=0,
    verbose=False,
):
    """
    Wrapper function for running DeepScence pipelines.

    Parameters
    ----------
    adata : AnnData
        AnnData object of the dataset where adata.X contains the expression count matrix.
    binarize : bool, optional, default=False
        Whether to binarize the output scores into SnCs vs. normal cells.
    species : str, optional, default="human"
        Species of the dataset, either "human" or "mouse".
    custome_gs : list of str, option, default=None
        If a custome gene set is desired, input it as a vector of gene symbols.
    anchor_gene: str, optional, default=None
        The encoded score may characterize senescence in the opposite direction. At default, the
        expression of CDKN1A/Cdkn1a is used to ensure correct direction. If CDKN1A/Cdkn1a is not
        in the dataset, set this parameter to a known positive senescence marker gene.
    lambda_ortho : float, optional, default=0.1
        Weight for the orthogonality regularization term.
    lambda_mmd : float, optional, default=0.7
        Weight for the Maximum Mean Discrepancy (MMD) regularization term, if `adata.obs["batch"]`
        is not present, this term will not be calculated.
    hidden_sizes : tuple, optional, default=(32,)
        Sizes of hidden layers for the encoder.
    batchnorm : bool, optional, default=False
        Whether to apply batch normalization to the neural network layers.
    dropout : float or None, optional, default=None
        Dropout rate for regularization. If None, no dropout is applied.
    epochs : int, optional, default=300
        Number of training epochs.
    validation_split : float, optional, default=0.1
        Fraction of cells to be used for validation during training.
    reduce_lr : int, optional, default=10
        Number of epochs to wait before reducing the learning rate if validation loss does not improve.
    early_stop : int, optional, default=25
        Number of epochs to wait before stopping training if validation loss does not improve.
    batch_size : int or None, optional, default=None
        Batch size for training. If None, a default batch size is used.
    learning_rate : float, optional, default=0.005
        Initial learning rate for the optimizer.
    n : int, optional, default=5
        Gene set membership threshold for genes.
    random_state : int, optional, default=0
        Seed for reproducibility.
    verbose : bool, optional, default=False
        Whether to display detailed logs.

    Returns
    -------
    AnnData
        AnnData object with senescence scores in `adata.obs["ds"]` and binarization results
        in `adata.obs["binary"]` if `binarize = True".

    Notes
    -----
    - The function assumes input data is properly filtered.
    - Setting `denoise = True` increases runtime, but is recommended.

    """
    assert isinstance(adata, anndata.AnnData), "adata must be an AnnData instance"
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

    # set seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = "0"
    torch.manual_seed(random_state)

    sc.pp.filter_genes(adata, min_cells=1)

    # coerce adata.X to csr
    if not sp.sparse.issparse(adata.X):
        adata.X = sp.sparse.csr_matrix(adata.X)

    # save the original version
    original = adata.copy()

    # read adata, subset, calculate up/down metrics
    adata = read_dataset(
        adata, species=species, n=n, custome_gs=custome_gs, verbose=True
    )
    if "b" not in adata.obs.columns:  # don't do MMD if no batch specified
        adata.obs["b"] = "placeholder"
        lambda_mmd = None

    input_size = adata.n_vars
    model = ZINBAutoencoder(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        size_factors=adata.obs["size_factors"],
        batchnorm=batchnorm,
        dropout=dropout,
        lambda_ortho=lambda_ortho,
        lambda_mmd=lambda_mmd,
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

    scores = model.predict(adata)

    scores, log = fix_score_direction(scores, adata, n, species, anchor_gene)
    original.obs["ds"] = scores
    original.uns["log"] = log

    if binarize:
        # use adata.obs["ds"] to fit a mixture of 2 normal.
        original = binarize_adata(original)

    original.X = sp.sparse.csr_matrix(original.X)
    return original
