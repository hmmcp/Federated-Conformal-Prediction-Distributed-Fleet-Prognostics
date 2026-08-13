# Federated Conformal Approach for Trustworthy Uncertainty Quantification in Distributed Fleet Prognostics

This repository provides the code used for the numerical studies presented in:

**Federated Conformal Approach for Trustworthy Uncertainty Quantification in Distributed Fleet Prognostics**

Manuscript:
https://www.techrxiv.org/doi/full/10.36227/techrxiv.176902673.30333692/v1


## Repository Overview

The repository contains numerical experiments using both real-world and simulated datasets to evaluate the proposed federated conformal prediction framework under heterogeneous and distribution-shifted environments.

The main experiments are organized as follows:

* **`C-MAPSS/main.py`**, together with **`functions.py`** and **`models.py`**, reproduces the experiments reported in **Section 5.3.1** of the manuscript using the **NASA C-MAPSS dataset**.
* **`Simulation/Simulation.py`** generates the synthetic experiments designed to evaluate the robustness of the proposed method under different forms of model misspecification and fleet heterogeneity.

---

## C-MAPSS: Remaining Useful Life Prediction

For the Remaining Useful Life (RUL) prediction task, we use a **Long Short-Term Memory (LSTM)** network as the backbone prediction model. **Quantile Regression (QR)** is used as the baseline uncertainty quantification method to construct prediction intervals.

We evaluate three uncertainty quantification approaches:

1. **No CP**:
   Prediction intervals are generated directly by the baseline QR model without conformal calibration.

2. **FCP**:
   Prediction intervals are calibrated using the baseline **Federated Conformal Prediction (FCP)** approach.

3. **Proposed Method**:
   Prediction intervals are calibrated using the proposed **Multilevel HMM-based Federated Conformal Prediction** approach.

The methods are compared in terms of:

* **Empirical coverage probability**, and
* **Prediction interval width**

under heterogeneous federated environments. A pre-trained model can be found in **`C-MAPSS/Models/QR_FD1_NL2_CSR0.2_C5_RS10000.pth`**. This model can be loaded to reproduce similar results. 

---

## ADNI Experiment

A detailed description of the features used in the ADNI experiment is provided in **`ADNI/Supplementary_Materials_ADNI_Features.csv`**. Because the ADNI dataset requires authorized access, the raw data cannot be publicly distributed with this repository. Researchers interested in accessing the dataset should submit a data access request through the official **Alzheimer's Disease Neuroimaging Initiative (ADNI)** website (https://adni.loni.usc.edu/).

The ADNI experiment follows the same general computational framework as the C-MAPSS experiment. Therefore, the code provided for the C-MAPSS dataset can be adapted to reproduce the ADNI analysis after obtaining the required ADNI data and performing the corresponding preprocessing.

---

## Simulation Study

The simulation experiments are implemented in: **`Simulation/Simulation.py`**


The simulation study is designed to investigate whether the proposed approach remains effective under different forms of model misspecification and heterogeneity.

We consider four experimental settings:

1. **Different numbers of HMM latent states**
2. **Violation of the left-to-right transition structure**
3. **Violation of the shared emission assumption**
4. **Fleet-level heterogeneity**

### Data Generation

We simulate a total of **5 fleets**, with **20 units per fleet**.

For each fleet, units are divided into:

* **70% training units**
* **30% calibration units**


In addition, a separate **test fleet containing 20 units** is generated. The test fleet follows a different transition matrix from the training fleets, introducing a controlled **distribution shift** between the training/calibration and test environments.

### Baseline Prediction Model

An **LSTM-based quantile regression model** is used as the baseline prediction model. The model is trained using the **pinball loss** to directly estimate lower and upper conditional quantiles.

The resulting QR intervals are then used as the basis for conformal calibration.

### Compared Methods

We compare three uncertainty quantification approaches:

1. **QR without CP**
   Prediction intervals are obtained directly from the quantile regression model without conformal calibration.

2. **FCP**
   Federated conformal prediction based on the partial exchangeability assumption.

3. **Proposed HMM-based FCP**
   The proposed method, which incorporates the latent HMM structure into the federated conformal calibration procedure.

To provide a more comprehensive evaluation, experiments are conducted under both:

* **Federated training**, and
* **Centralized training**

settings.

### Evaluation

For each experimental configuration, the complete experiment is repeated **10 times**, and the reported results are averaged across repetitions.

The main evaluation metrics are:

* **Empirical coverage probability**
* **Average prediction interval width**

---

## Reproducibility

In  **`Simulation/Simulation.py`**, by default, the hyperparameter:

```python
N_REPS = 10
```

controls the number of independent experimental repetitions.

The results reported in the manuscript are obtained using:

```python
N_REPS = 10
```

For a quick reproducibility check or to verify that the code runs correctly on your system, you may reduce this value, for example:

```python
N_REPS = 1
```

or

```python
N_REPS = 2
```

Reducing `N_REPS` can substantially decrease the computational cost, but the resulting numerical values may differ from those reported in the manuscript because fewer independent repetitions are averaged.

---

```
