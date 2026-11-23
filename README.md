# Simulated Hospital Federated Learning Prototype (Breast Cancer Benchmark)

> **Important Data Disclaimer:** All hospital sites (`Hospital_Sim_1` – `Hospital_Sim_4`) in this project are **simulated** by splitting the public `sklearn.datasets.load_breast_cancer()` benchmark dataset. **No real patient or clinical hospital records are used**, and this prototype is for educational and portfolio demonstration only—not clinical diagnosis.

## Problem & Goal

Healthcare organizations often cannot pool raw patient records centrally due to privacy regulations, institutional governance, and data-sharing barriers. The goal of this prototype is to implement **Federated Averaging (FedAvg)** from scratch in `numpy` across **4 simulated hospital sites** without sharing raw patient rows, and compare its performance against:

1. **Centralized training** (logistic regression on pooled training data).
2. **Local-only training** (each simulated site training independently on its own split).

We evaluate both **IID** (unequal random splits) and **Non-IID** (unequal splits with skewed malignant/benign class balance) partitions on a held-out **20% stratified test set**.

## Method

* **Train / Test Split (`SEED = 42`):** Stratified 80/20 split of `sklearn.datasets.load_breast_cancer()` (`569` samples, `30` numeric features), keeping the 20% test set strictly untouched during training. In scikit-learn's dataset, `target == 0` is **malignant** and `target == 1` is **benign**.
* **Simulated Hospital Partitions (4 Sites):**
  * **IID Split:** Random permutation of the training set divided into unequal site sizes (`35%`, `25%`, `25%`, `15%`).
  * **Non-IID Split:** Unequal site sizes with skewed malignant/benign ratios across sites (e.g., heavily malignant vs. heavily benign splits), including edge cases where a single-class local split can occur.
* **Privacy-Aware Standardization:** Instead of fitting a scaler on pooled raw patient records, global mean and standard deviation are computed from per-site sample counts, feature sums, and sums of squared deviations around the global mean (summary moments only).
* **Hand-Written FedAvg (`numpy`):** Over `25` communication rounds (`5` local gradient-descent epochs per round, L2 regularization, learning rate `0.15`), each simulated hospital trains a binary logistic regression model starting from the current global weights `(w, b)`. The server aggregates model parameters weighted by each site's sample count.
* **Optional Gaussian Weight Noise Experiment:** Zero-mean Gaussian noise ($\mathcal{N}(0, \sigma^2)$) is added to aggregated weights across varying noise levels ($\sigma$) to illustrate the trade-off between weight perturbation and predictive utility.

## Results

*(Replace the bracketed values below with the exact metrics printed by your notebook run.)*

| Model / Setup | Data Split | Test Accuracy | Malignant Recall | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| Centralized Logistic Regression | Pooled Train | `[paste acc]` | `[paste recall]` | `[paste auc]` |
| FedAvg (25 rounds, 5 local epochs) | IID (Simulated) | `[paste acc]` | `[paste recall]` | `[paste auc]` |
| FedAvg (25 rounds, 5 local epochs) | Non-IID (Simulated) | `[paste acc]` | `[paste recall]` | `[paste auc]` |
| Local-Only — `Hospital_Sim_1` | IID (Simulated) | `[paste acc]` | `[paste recall]` | `[paste auc]` |
| Local-Only — `Hospital_Sim_2` | IID (Simulated) | `[paste acc]` | `[paste recall]` | `[paste auc]` |
| Local-Only — `Hospital_Sim_3` | IID (Simulated) | `[paste acc]` | `[paste recall]` | `[paste auc]` |
| Local-Only — `Hospital_Sim_4` | IID (Simulated) | `[paste acc]` | `[paste recall]` | `[paste auc]` |
| Local-Only — `Hospital_Sim_1` | Non-IID (Simulated) | `[paste acc]` | `[paste recall]` | `[paste auc]` |
| Local-Only — `Hospital_Sim_2` | Non-IID (Simulated) | `[paste acc]` | `[paste recall]` | `[paste auc]` |
| Local-Only — `Hospital_Sim_3` | Non-IID (Simulated) | `[paste acc]` | `[paste recall]` | `[paste auc]` |
| Local-Only — `Hospital_Sim_4` | Non-IID (Simulated) | `[paste acc]` | `[paste recall]` | `[paste auc]` |

### Generated Figures

* `figures/iid_vs_noniid_round_accuracy.png` — Hold-out test accuracy across federated rounds for IID vs. Non-IID simulated hospitals compared to the centralized baseline.
* `figures/model_comparison_metrics.png` — Comparison of Accuracy, Malignant Recall, and ROC-AUC across centralized, federated, and local-only simulated sites.
* `figures/dp_noise_tradeoff.png` — Effect of adding Gaussian noise ($\sigma$) to shared weights on test accuracy, malignant recall, and ROC-AUC.

## Key Takeaways & Remaining Privacy Risks

* **When FedAvg matches centralized training:** Under IID partitions with a convex/smooth model like logistic regression and well-chosen learning rate and local epoch count, FedAvg closely tracks centralized performance.
* **What is lost under Non-IID data:** Skewed class distributions cause local updates to drift in conflicting directions ("client drift"), which can slow convergence, reduce overall accuracy, or hurt recall on the malignant class—where missed malignant cases (false negatives) are critical.
* **Privacy risks that remain:** Sharing model parameters or summary statistics instead of raw patient tables **does not guarantee privacy**. Updates can still leak sensitive information via gradient/model inversion, membership inference, or class-distribution cues. Plain FedAvg lacks formal differential privacy guarantees or secure aggregation unless combined with gradient clipping, calibrated noise (e.g., DP-SGD), and cryptographic aggregation.

## Limitations

* **Simulated dataset & partitions:** The 4 hospitals are synthetic splits of a single public dataset (`569` rows total), not real multi-center clinical cohorts with domain shift, missingness, or varying EHR schemas.
* **Simple linear model:** Logistic regression is fast and transparent for a prototype, but non-linear clinical models (e.g., neural networks or gradient-boosted trees) behave differently under non-IID skew.
* **Heuristic noise vs. formal DP:** Adding Gaussian noise to aggregated weights demonstrates a qualitative privacy–utility trade-off, not a formal $(\varepsilon, \delta)$-differential privacy bound.

## How to Run

1. Open `federated_breast_cancer.ipynb` in Jupyter Notebook or Google Colab.
2. Run all cells in order (`Shift + Enter`).
3. Inspect the printed metrics and generated plots in `figures/`.
