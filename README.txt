This code corresponds to the numerical results in Section 4 of the manuscript 'Federated Conformal Approach for Trustworthy Uncertainty Quantification in Distributed Fleet Prognostics.'

By default, N_REPS is set to 10. For a quick reproducibility check or to verify that the code runs correctly on your system, you may reduce N_REPS to a smaller value (e.g., 1 or 2).

We design four experimental settings to examine whether the proposed approach remains effective under various forms of model misspecification and heterogeneity: (i) different numbers of HMM latent states, (ii) violations of the left-to-right transition structure, (iii) violations of the shared emission assumption, and (iv) fleet-level heterogeneity. 

We simulate a total of 5 fleets, each fleet contains 20 units. For each fleet, data are split into training and calibration subsets (70% / 30%) at the unit level. A separate test fleet of 20 units is generated with a different transition matrix to induce a distribution shift.

An LSTM-based quantile regression (QR) model using the pinball loss is trained as baseline model for Conformal prediction. We compare three approaches: (1) QR without CP, (2) FCP based on partial exchangeability, and (3) the proposed HMM-based CP method. To provide a more comprehensive evaluation, both federated and centralized training settings are considered. For all results reported, we iterate the model for 10 times and report the average coverage and width.
