## 🧬 Single-cell and spatial detection of senescent cells using DeepScence

**DeepScence** is an unsupervised machine learning model based on autoencoders for accurately scoring and identifying senescent cells in single-cell RNA-seq (scRNA-seq) and spatial transcriptomics datasets.

![DeepScence Overview](DeepScence/data/DeepScence_overview.png)

DeepScence takes as input an **`AnnData`** object with denoised expression count matrix stored in `adata.X` and outputs:
- A continuous senescence score (`adata.obs["ds"]`)
- A binary classification (`adata.obs["binary"]`)

---

### 📦 Installation

We recommend having denoised gene expression count matrix in `adata.X` using **[DCA](https://github.com/theislab/dca)** before running DeepScence. However, The original DCA depends on:

```
tensorflow >= 2.0, < 2.5
keras >= 2.4, < 2.6
```
These versions conflict with DeepScence’s dependencies and do not run natively on Apple silicon. Therefore we recommend the following installtion workflow using our **patched DCA** (compatible with **TensorFlow 2.12.x**) for a smooth setup.

---

#### 💻 For x86-64 systems (Intel/AMD CPUs; Linux, Windows, macOS-Intel)

Starting from a clean conda environment:
```bash
conda create -n deepscence python=3.8
conda activate deepscence
```

Install DeepScence and dependencies:
```bash
pip install git+https://github.com/anthony-qu/DeepScence.git
pip install tensorflow==2.12.*
pip install git+https://github.com/anthony-qu/dca_patched.git
```

---

#### 🍏 For Apple silicon (M1/M2/M3, ARM64 macOS)

> **Note:** The original TensorFlow requires AVX CPU instructions (not supported on Apple silicon). Use `tensorflow-macos`instead.

Starting from a clean conda environment:
```bash
CONDA_SUBDIR=osx-arm64 conda create -n deepscence python=3.8
conda activate deepscence
python -c "import platform; print(platform.machine())"  # should print: arm64
```

Install DeepScence and dependencies:
```bash
pip install git+https://github.com/anthony-qu/DeepScence.git
pip install "numpy>=1.22,<1.24" "tensorflow-macos==2.12.0"
pip install git+https://github.com/anthony-qu/dca_patched.git
```

---


### 📖 Tutorial

A step-by-step usage example of DeepScence is available here:

[**View the tutorial notebook**](DeepScence/tutorial/demo.ipynb)
