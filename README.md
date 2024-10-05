## Single-cell and spatial detection of senescent cells using DeepScence.

DeepScence: an unsupervised machine learning model based on autoencoders to accurately score and identify senescent cells in single-cell RNA-seq (scRNA-seq) and spatial transcriptomics datasets. DeepScence takes as input a `Anndata` object with expression count matrix store at `adata.X`. DeepScence outputs a continuous score and classification of senescent cells by adding columns `["deepscence", "binary"]` to `adata.obs`. 

### Installation

#### pip

Python installation of DeepScence and its dependencies:

```
$ pip install DeepScence
```


### Usage

In python, for scoring only:

```
import scanpy as sc
from DeepScence.api import DeepScence

adata = sc.read_h5ad("/path/to/h5ad/")
adata_scored = DeepScence(adata)
```

if senescence binarization is needed, call:

```
adata_scored = DeepScence(adata, binarize=True)
```

The resulting `adata_scored` contains senescence scores in `adata_scored.obs["ds"]`, and binarization result (1: senescent cells, 0: normal cells) in `adata_scored.obs["binary"]` if `binarize=True`. 