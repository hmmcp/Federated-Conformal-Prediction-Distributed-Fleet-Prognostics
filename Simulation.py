#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation Study 
  A) Sensitivity to M (true M*=4, fit M in {2,3,4,5,7})
  B) Violation of left-to-right (partial recovery)
  C) Violation of shared-emission (fleet-specific shifts)
  D) Fleet heterogeneity in transition dynamics

Each with: No CP / Lu et al. FCP / Proposed, both Federated & Centralized.
"""

import numpy as np
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
from tdigest import TDigest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

os.makedirs(r"C:\Users\dogha\Desktop\HMM_simulation\Simulation_Results\Models", exist_ok=True)
os.makedirs(r"C:\Users\dogha\Desktop\HMM_simulation\Simulation_Results\Outputs", exist_ok=True)
os.makedirs(r"C:\Users\dogha\Desktop\HMM_simulation\Simulation_Results\Diagnostics", exist_ok=True)

########################################
# GLOBAL SETTINGS
########################################
ALPHA = 0.1
PREDICTION_HORIZON = 10 # To predict signal at t+PREDICTION_HORIZON
N_SENSORS = 5
SEQ_LEN = 20
BATCH_SIZE = 32
Tcw = 10

"""
Training quantiles to train QR model
Option 1 is to normal test. 
Option 2 is to stress test CP methods. We set qauntiles intentionally tighter than the target coverage to esnure the base model undercovers
"""
#TRAIN_QUANTILES = [ALPHA/2, 0.5, 1-ALPHA/2] # Option 1: Normal test
TRAIN_QUANTILES = [0.15, 0.5, 0.85] # Option 2: Stress test with deliverately imperfect baseline UQ


########################################
# SECTION 1-1: SYNTHETIC DATA GENERATION
########################################

class SyntheticDGP:
    """
    Synthetic Data Generating Process (DGP)
    """

    def __init__(
        self,
        num_true_states=4,
        num_fleets=5,
        units_per_fleet=None,
        trajectory_range=(80, 150),
        dirichlet_concentration=50.0, # Larger value = Larger similarity in degradation transition
        emission_means=None,
        emission_stds=None,
        fleet_emission_shift=0.0,
        reverse_prob=0.0,
        base_transition_stay=0.96, 
        signal_momentum=0.7, # To temporally smooth signals (x_t = momentum * x_{t-1} + (1-momentum) * (mean_m + noise))
        random_state=42,
    ):
        self.rng = np.random.default_rng(random_state)
        self.M_true = num_true_states
        self.K = num_fleets
        self.units_per_fleet = units_per_fleet or [20] * num_fleets
        self.traj_range = trajectory_range
        self.dirichlet_conc = dirichlet_concentration
        self.reverse_prob = reverse_prob
        self.base_stay = base_transition_stay
        self.signal_momentum = signal_momentum
        self.n_sensors = N_SENSORS

        if emission_means is None:
            self.emission_means = np.zeros((self.M_true, self.n_sensors))
            for s in range(self.n_sensors):
                base_offset = 0.05 * s  # This is for sensor-to-sensor offset
                for m in range(self.M_true):
                    # To separate different stages
                    self.emission_means[m, s] = 0.1 + 0.35 * m + 0.05 * m**2 + base_offset
        else:
            self.emission_means = emission_means

        if emission_stds is None:
            # To increase noise as degradation proceeds
            self.emission_stds = np.zeros((self.M_true, self.n_sensors))
            for m in range(self.M_true):
                self.emission_stds[m, :] = 0.03 + 0.06 * m  
        else:
            self.emission_stds = emission_stds

        # Fleet-specific emission shifts
        self.fleet_shifts = np.zeros((self.K, self.n_sensors))
        if fleet_emission_shift > 0:
            self.fleet_shifts = self.rng.uniform(
                -fleet_emission_shift, fleet_emission_shift,
                size=(self.K, self.n_sensors)
            )

        # To generate fleet-specific transition matrices
        self.fleet_A = self._generate_fleet_transitions()

        self.true_params = {
            'M_true': self.M_true,
            'fleet_A': self.fleet_A,
            'emission_means': self.emission_means,
            'emission_stds': self.emission_stds,
            'fleet_shifts': self.fleet_shifts,
            'reverse_prob': self.reverse_prob,
        }

    def _generate_fleet_transitions(self):
        """
        Generate fleet-specific transition matrices.
        """
        fleet_A = []
        for k in range(self.K):
            A = np.zeros((self.M_true, self.M_true))
            for m in range(self.M_true):
                if m == self.M_true - 1:
                    A[m, m] = 1.0  # absorbing
                else:
                    conc_stay = self.dirichlet_conc * self.base_stay
                    conc_advance = self.dirichlet_conc * (1.0 - self.base_stay)
                    probs = self.rng.dirichlet([conc_stay, conc_advance])
                    A[m, m] = probs[0]
                    A[m, m + 1] = probs[1]

                    #'''
                    if self.reverse_prob > 0 and m > 0:
                        p_back = min(self.reverse_prob, A[m, m] * 0.5)
                        A[m, m - 1] = p_back
                        A[m, m] -= p_back
                        row_sum = A[m].sum()
                        if row_sum > 0:
                            A[m] /= row_sum
                    #'''        
            fleet_A.append(A)
        return fleet_A

    def generate_unit(self, fleet_id):
        """
        Generate degradation signals, degradation stages, and lifetime of one unit
        """
        T = self.rng.integers(self.traj_range[0], self.traj_range[1] + 1)
        A = self.fleet_A[fleet_id]

        states = np.zeros(T, dtype=int)
        states[0] = 0

        for t in range(1, T):
            states[t] = self.rng.choice(self.M_true, p=A[states[t - 1]])

        signals = np.zeros((T, self.n_sensors))
        # Initialize first step
        m0 = states[0]
        mean0 = self.emission_means[m0] + self.fleet_shifts[fleet_id]
        std0 = self.emission_stds[m0]
        signals[0] = self.rng.normal(mean0, std0)

        for t in range(1, T):
            m = states[t]
            target = self.emission_means[m] + self.fleet_shifts[fleet_id]
            noise = self.rng.normal(0, self.emission_stds[m])
            # Momentum-based smoothing
            signals[t] = (self.signal_momentum * signals[t - 1] +
                          (1 - self.signal_momentum) * (target + noise))

        return signals, states, T

    def generate_dataset(self):
        fleet_data = []
        unit_counter = 0
        for k in range(self.K):
            units = []
            for j in range(self.units_per_fleet[k]):
                signals, states, T = self.generate_unit(k)
                units.append({
                    'signals': signals,
                    'states': states,
                    'T': T,
                    'fleet_id': k,
                    'unit_id': unit_counter,
                })
                unit_counter += 1
            fleet_data.append(units)
        return fleet_data

    def generate_test_fleet(self, num_units=50, fleet_transition=None,
                            degradation_speed_factor=1.0):
        """
        Generate a target (test) fleet. This fleet has a different degradation dynamics from the training fleets..
        """
        if fleet_transition is None:
            fleet_transition = np.mean(self.fleet_A, axis=0).copy()
        
        #TODO: Test fleet noise multiplier for extra challenge?
        test_noise_mult = 1.0

        test_units = []
        for j in range(num_units):
            T = self.rng.integers(self.traj_range[0], self.traj_range[1] + 1)
            states = np.zeros(T, dtype=int)
            states[0] = 0
            for t in range(1, T):
                states[t] = self.rng.choice(self.M_true, p=fleet_transition[states[t - 1]])

            signals = np.zeros((T, self.n_sensors))
            m0 = states[0]
            signals[0] = self.rng.normal(self.emission_means[m0],
                                         self.emission_stds[m0] * test_noise_mult)
            for t in range(1, T):
                m = states[t]
                target = self.emission_means[m]
                noise = self.rng.normal(0, self.emission_stds[m] * test_noise_mult)
                signals[t] = (self.signal_momentum * signals[t - 1] +
                              (1 - self.signal_momentum) * (target + noise))

            test_units.append({
                'signals': signals,
                'states': states,
                'T': T,
                'fleet_id': self.K,
                'unit_id': j,
            })
        return test_units


########################################
# SECTION 1-2: VISUALIZATION OF SYNTHETIC DATA
########################################

def plot_diagnostics(dgp, fleet_data, tag="base"):
    """
    Generate plots to verify generated synthetic data
    """
    fig_dir = r"C:\Users\dogha\Desktop\HMM_simulation\Simulation_Results\Diagnostics"

    fig, axes = plt.subplots(dgp.K, 1, figsize=(14, 3 * dgp.K), sharex=False)
    if dgp.K == 1:
        axes = [axes]
    for k in range(dgp.K):
        ax = axes[k]
        for j, unit in enumerate(fleet_data[k][:3]):
            ax.plot(unit['signals'][:, 0], label=f"unit {unit['unit_id']}", alpha=0.8)
        ax.set_title(f"Fleet {k+1}: sample unit trajectories (sensor 0)", fontsize=11)
        ax.set_ylabel("signal")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.2, 2.0)
    axes[-1].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/sample_units_sensor0_{tag}.png", dpi=150)
    plt.close()

    stage_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    n_show = min(2, len(fleet_data[0]))
    n_fleets_show = min(3, dgp.K)
    fig, axes = plt.subplots(n_show * n_fleets_show, 1,
                             figsize=(14, 2.5 * n_show * n_fleets_show))
    axes = np.atleast_1d(axes)
    idx = 0
    for k in range(n_fleets_show):
        for j in range(n_show):
            if j >= len(fleet_data[k]):
                continue
            unit = fleet_data[k][j]
            ax = axes[idx]
            states = unit['states']
            sig = unit['signals'][:, 0]
            for t in range(len(sig) - 1):
                ax.plot([t, t + 1], [sig[t], sig[t + 1]],
                        color=stage_colors[states[t] % len(stage_colors)], lw=1.5)
            ax.set_title(f"Fleet {k+1}, Unit {unit['unit_id']} (sensor 0, stage-colored)",
                         fontsize=10)
            ax.set_ylabel("signal")
            ax.set_ylim(-0.2, 2.0)
            idx += 1
    axes[-1].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/stage_colored_sensor0_{tag}.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    occupancy = np.zeros((dgp.K, dgp.M_true))
    for k in range(dgp.K):
        for unit in fleet_data[k]:
            for m in range(dgp.M_true):
                occupancy[k, m] += np.sum(unit['states'] == m)
    # Normalize per fleet
    fleet_totals = occupancy.sum(axis=1, keepdims=True)
    fleet_totals[fleet_totals == 0] = 1
    occupancy_frac = occupancy / fleet_totals

    bottom = np.zeros(dgp.K)
    for m in range(dgp.M_true):
        ax.bar(range(dgp.K), occupancy_frac[:, m], bottom=bottom,
               label=f"stage {m}", color=stage_colors[m % len(stage_colors)])
        bottom += occupancy_frac[:, m]
    ax.set_xticks(range(dgp.K))
    ax.set_xticklabels([f"Fleet {k+1}" for k in range(dgp.K)])
    ax.set_ylabel("proportion of time")
    ax.set_title("True stage occupancy by fleet")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/stage_occupancy_by_fleet_{tag}.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(dgp.K):
        means = []
        stds = []
        for m in range(dgp.M_true):
            vals = []
            for unit in fleet_data[k]:
                mask = unit['states'] == m
                if mask.sum() > 0:
                    vals.extend(unit['signals'][mask, 0].tolist())
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(np.nan)
                stds.append(0)
        means = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(range(dgp.M_true), means, 'o-', label=f"Fleet {k+1}")
        ax.fill_between(range(dgp.M_true), means - stds_arr, means + stds_arr, alpha=0.1)
    ax.set_xlabel("true latent stage")
    ax.set_ylabel("sensor 0 value")
    ax.set_title("Fleet-wise mean signal by true stage")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/stage_means_sensor0_{tag}.png", dpi=150)
    plt.close()

    # Print summary stats
    print(f"  Stage occupancy (fleet-averaged): "
          f"{np.mean(occupancy_frac, axis=0).round(3)}")
    print(f"  Emission means (sensor 0): {dgp.emission_means[:, 0].round(3)}")
    print(f"  Emission stds (sensor 0):  {dgp.emission_stds[:, 0].round(3)}")
    for k in range(dgp.K):
        diag = np.diag(dgp.fleet_A[k])
        print(f"  Fleet {k+1} A diagonal: {diag.round(4)}")


########################################
# SECTION 2: PREDICTION MODEL
########################################

class QuantileLSTM(nn.Module):
    def __init__(self, input_size=N_SENSORS, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def pinball_loss(output, target, quantiles=None):
    if quantiles is None:
        quantiles = [ALPHA / 2, 0.5, 1 - ALPHA / 2]
    target = target.view(-1, 1)
    total_loss = 0.0
    for i, q in enumerate(quantiles):
        pred = output[:, i].view(-1, 1)
        error = target - pred
        loss = torch.mean(torch.max(q * error, (q - 1) * error))
        total_loss += loss
    return total_loss / len(quantiles)


class SignalForecastDataset(Dataset):
    def __init__(self, units, seq_len=SEQ_LEN, horizon=PREDICTION_HORIZON, stride=1):
        self.data = []
        for unit in units:
            signals = unit['signals']
            T = unit['T']
            for t in range(0, T - seq_len - horizon + 1, stride):
                x = signals[t:t + seq_len]
                y = signals[t + seq_len - 1 + horizon, 0]
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


########################################
# SECTION 3: MODEL TRAINING (SAME IN BOTH SIMULATION AND CASE STUDIES)
########################################

def train_model_centralized(model, train_units, seq_len=SEQ_LEN, horizon=PREDICTION_HORIZON,
                            batch_size=BATCH_SIZE, epochs=30, lr=0.002):
    dataset = SignalForecastDataset(train_units, seq_len=seq_len, horizon=horizon)
    if len(dataset) == 0:
        return model
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = pinball_loss(out, y, quantiles=TRAIN_QUANTILES)
            loss.backward()
            optimizer.step()
    return model


def train_model_federated(model, fleet_train_data, batch_size=BATCH_SIZE,
                          epochs_local=5, lr=0.002, rounds=15,
                          seq_len=SEQ_LEN, horizon=PREDICTION_HORIZON):
    device = torch.device("cpu")
    global_model = model.to(device)
    global_model.train()

    fleet_sizes = []
    for fleet_units in fleet_train_data:
        ds = SignalForecastDataset(fleet_units, seq_len=seq_len, horizon=horizon)
        fleet_sizes.append(len(ds))
    total_size = sum(fleet_sizes)
    if total_size == 0:
        return global_model

    for r in range(rounds):
        local_state_dicts = []
        local_weights = []

        for k, fleet_units in enumerate(fleet_train_data):
            ds = SignalForecastDataset(fleet_units, seq_len=seq_len, horizon=horizon)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            local_model = QuantileLSTM(input_size=N_SENSORS,
                                       hidden_size=global_model.hidden_size,
                                       num_layers=global_model.num_layers).to(device)
            local_model.load_state_dict(global_model.state_dict())
            local_model.train()

            optimizer = optim.Adam(local_model.parameters(), lr=lr)
            for epoch in range(epochs_local):
                for x, y in loader:
                    optimizer.zero_grad()
                    out = local_model(x)
                    loss = pinball_loss(out, y, quantiles=TRAIN_QUANTILES)
                    loss.backward()
                    optimizer.step()

            local_state_dicts.append(local_model.state_dict())
            local_weights.append(fleet_sizes[k])

        if not local_state_dicts:
            continue

        total_w = sum(local_weights)
        global_state_dict = {}
        for key in global_model.state_dict().keys():
            weighted_sum = sum(
                w * sd[key].float() for w, sd in zip(local_weights, local_state_dicts)
            )
            global_state_dict[key] = weighted_sum / total_w
        global_model.load_state_dict(global_state_dict)

    return global_model


########################################
# SECTION 4: CONFORMAL PREDICTION (SAME IN BOTH SIMULATION AND CASE STUDIES)
########################################

class FederatedHMM:
    def __init__(self, num_states, initial_scores, a_diag=0.8, zeta_val=10.0, random_state=None):
        rng = np.random.default_rng(random_state)
        self.num_states = num_states
        self.pi = rng.dirichlet(alpha=np.ones(num_states))
        self.emission_digests = [TDigest() for _ in range(num_states)]
        if initial_scores:
            for i, score in enumerate(initial_scores):
                self.emission_digests[i % num_states].update(score)
        self.emission_pdfs = [self._build_pdf_from_digest(d) for d in self.emission_digests]
        self.A_init = np.zeros((num_states, num_states))
        for i in range(num_states - 1):
            self.A_init[i, i] = a_diag
            self.A_init[i, i + 1] = 1.0 - a_diag
        self.A_init[num_states - 1, num_states - 1] = 1.0
        self.zeta = np.full((num_states, 2), zeta_val)

    def _build_pdf_from_digest(self, digest):
        if digest.n == 0:
            return [], []
        centroids = sorted(digest.centroids_to_list(), key=lambda c: c['m'])
        total_weight = sum(c['c'] for c in centroids)
        if total_weight == 0:
            return [], []
        means = [c['m'] for c in centroids]
        probs = [c['c'] / total_weight for c in centroids]
        boundaries = [(means[i] + means[i + 1]) / 2 for i in range(len(means) - 1)]
        return probs, boundaries

    def _get_emission_likelihood_from_pdfs(self, score, state_idx, emission_pdfs):
        probs, boundaries = emission_pdfs[state_idx]
        if not probs:
            return 1e-9
        j = np.searchsorted(boundaries, score)
        return max(probs[j], 1e-9)

    def get_emission_likelihood(self, score, state_idx):
        return self._get_emission_likelihood_from_pdfs(score, state_idx, self.emission_pdfs)

    def _approx_emission_Q(self, digests, emission_pdfs):
        Q = 0.0
        for m, digest in enumerate(digests):
            if digest.n == 0:
                continue
            for c in digest.centroids_to_list():
                score, weight = c['m'], c['c']
                p = self._get_emission_likelihood_from_pdfs(score, m, emission_pdfs)
                Q += weight * np.log(max(p, 1e-9))
        return Q

    def global_m_step(self, client_statistics, all_client_digests, q_tolerance=1e-9):
        if not client_statistics:
            return
        total_pi_updates = np.sum([s['pi_update'] for s in client_statistics], axis=0)
        if np.sum(total_pi_updates) > 0:
            self.pi = total_pi_updates / np.sum(total_pi_updates)
        old_emission_pdfs = self.emission_pdfs
        new_emission_digests = [TDigest() for _ in range(self.num_states)]
        for client_digests in all_client_digests:
            if not client_digests:
                continue
            for m in range(self.num_states):
                if m < len(client_digests) and client_digests[m] is not None:
                    new_emission_digests[m] += client_digests[m]
        candidate_emission_pdfs = [self._build_pdf_from_digest(d) for d in new_emission_digests]
        Q_old = self._approx_emission_Q(new_emission_digests, old_emission_pdfs)
        Q_new = self._approx_emission_Q(new_emission_digests, candidate_emission_pdfs)
        if Q_new >= Q_old - q_tolerance:
            self.emission_digests = new_emission_digests
            self.emission_pdfs = candidate_emission_pdfs


def _calculate_hmm_posteriors(client_scores, pi, A, emission_likelihood_func):
    n_obs = len(client_scores)
    num_states = A.shape[0]
    if n_obs == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0
    fwd = np.zeros((n_obs, num_states))
    scaling_factors = np.zeros(n_obs)
    ep0 = np.array([emission_likelihood_func(client_scores[0], m) for m in range(num_states)])
    fwd[0, :] = pi * ep0
    scaling_factors[0] = np.sum(fwd[0, :])
    if scaling_factors[0] > 0:
        fwd[0, :] /= scaling_factors[0]
    for t in range(1, n_obs):
        ep = np.array([emission_likelihood_func(client_scores[t], m) for m in range(num_states)])
        fwd[t, :] = np.dot(fwd[t - 1, :], A) * ep
        scaling_factors[t] = np.sum(fwd[t, :])
        if scaling_factors[t] > 0:
            fwd[t, :] /= scaling_factors[t]
    bwd = np.zeros((n_obs, num_states))
    bwd[n_obs - 1, :] = 1.0
    for t in range(n_obs - 2, -1, -1):
        ep = np.array([emission_likelihood_func(client_scores[t + 1], m) for m in range(num_states)])
        if scaling_factors[t + 1] > 0:
            bwd[t, :] = np.dot(A, (ep * bwd[t + 1, :])) / scaling_factors[t + 1]
        else:
            bwd[t, :] = 1.0 / num_states
    eta = fwd * bwd
    row_sums = eta.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    eta /= row_sums
    log_likelihood = np.sum(np.log(scaling_factors + 1e-9))
    return eta, fwd, bwd, scaling_factors, log_likelihood

def run_local_e_step(client_scores, global_pi, local_A, emission_likelihood_func):
    n_obs = len(client_scores)
    num_states = local_A.shape[0]
    eta, fwd, bwd, scaling_factors, log_likelihood = _calculate_hmm_posteriors(
        client_scores, global_pi, local_A, emission_likelihood_func
    )
    if n_obs == 0:
        return {'pi_update': np.zeros(num_states),
                'A_transitions': np.zeros((num_states, num_states))}, \
               [TDigest() for _ in range(num_states)], 0
    xi = np.zeros((n_obs - 1, num_states, num_states))
    for t in range(n_obs - 1):
        ep = np.array([emission_likelihood_func(client_scores[t + 1], m) for m in range(num_states)])
        denom = np.dot(np.dot(fwd[t, :].T, local_A) * ep, bwd[t + 1, :])
        if denom > 0:
            for i in range(num_states):
                xi[t, i, :] = fwd[t, i] * local_A[i, :] * ep * bwd[t + 1, :] / denom
    pi_update = eta[0, :]
    A_transitions = np.sum(xi, axis=0)
    digest_values = client_scores
    client_state_digests = [TDigest() for _ in range(num_states)]
    for t in range(n_obs):
        for m in range(num_states):
            if eta[t, m] > 1e-9:
                client_state_digests[m].update(digest_values[t], eta[t, m])
    return {'pi_update': pi_update, 'A_transitions': A_transitions}, \
           client_state_digests, log_likelihood

def run_local_m_step(local_A_transitions, zeta):
    num_states = local_A_transitions.shape[0]
    A_new = np.zeros((num_states, num_states))
    for m in range(num_states - 1):
        N_mm = local_A_transitions[m, m]
        N_m_m1 = local_A_transitions[m, m + 1]
        zeta_mm = zeta[m, 0]
        zeta_m_m1 = zeta[m, 1]
        denom = (N_mm + N_m_m1) + (zeta_mm + zeta_m_m1) - 2
        if denom > 1e-9:
            A_new[m, m] = (N_mm + zeta_mm - 1) / denom
            A_new[m, m + 1] = (N_m_m1 + zeta_m_m1 - 1) / denom
        else:
            A_new[m, m] = A_new[m, m + 1] = 0.5
    A_new[num_states - 1, num_states - 1] = 1.0
    return A_new


def predict_state_weights(score_context, global_pi, local_A, emission_likelihood_func):
    n_context = len(score_context)
    num_states = local_A.shape[0]
    if n_context == 0:
        return global_pi
    fwd = np.zeros((n_context, num_states))
    ep0 = np.array([emission_likelihood_func(score_context[0], m) for m in range(num_states)])
    fwd[0, :] = global_pi * ep0
    s = np.sum(fwd[0, :])
    if s > 0:
        fwd[0, :] /= s
    for t in range(1, n_context):
        ep = np.array([emission_likelihood_func(score_context[t], m) for m in range(num_states)])
        fwd[t, :] = np.dot(fwd[t - 1, :], local_A) * ep
        s = np.sum(fwd[t, :])
        if s > 0:
            fwd[t, :] /= s
    return fwd[-1, :]


def get_likelihood_ratio_weights(score_context, client_pdfs, global_pdf):
    num_clients = len(client_pdfs)
    log_likelihoods_k = np.zeros(num_clients)
    log_likelihood_global = 0.0

    def get_prob_from_pdf(score, pdf):
        probs, boundaries = pdf
        if not probs:
            return 1e-9
        j = np.searchsorted(boundaries, score)
        return max(probs[j], 1e-9)

    for score in score_context:
        log_likelihood_global += np.log(get_prob_from_pdf(score, global_pdf))
        for k in range(num_clients):
            log_likelihoods_k[k] += np.log(get_prob_from_pdf(score, client_pdfs[k]))
    log_ratios = log_likelihoods_k - log_likelihood_global
    ratios = np.exp(log_ratios - np.max(log_ratios))
    return ratios / np.sum(ratios) if np.sum(ratios) > 0 else np.ones(num_clients) / num_clients


def calculate_global_average_A(local_A_matrices):
    return np.sum(local_A_matrices, axis=0) / len(local_A_matrices)


def get_adaptive_quantile(rho_m, omega_k, all_client_digests, alpha):
    final_digest = TDigest()
    num_clients = len(all_client_digests)
    num_states = len(rho_m)
    for k in range(num_clients):
        if not all_client_digests[k]:
            continue
        for m in range(num_states):
            weight = rho_m[m] * omega_k[k]
            if weight > 1e-9:
                scaled_digest = TDigest()
                try:
                    for centroid in all_client_digests[k][m].centroids_to_list():
                        scaled_digest.update(centroid['m'], centroid['c'] * weight)
                    final_digest += scaled_digest
                except (ValueError, IndexError):
                    continue
    if final_digest.n == 0:
        return np.inf
    return final_digest.percentile(100 * (1 - alpha))

def run_federated_hmm_training(NUM_HMM_STATES, NUM_EM_ROUNDS, CONVERGENCE_THRESHOLD,
                               client_score_sequences, all_cal_scores,
                               a_diag=0.8, zeta_val=10.0, random_seed=0):
    hmm = FederatedHMM(num_states=NUM_HMM_STATES, initial_scores=all_cal_scores,
                       a_diag=a_diag, zeta_val=zeta_val, random_state=random_seed)
    num_clients = len(client_score_sequences)
    local_A_matrices = [hmm.A_init.copy() for _ in range(num_clients)]
    all_client_per_state_digests = [[] for _ in range(num_clients)]
    prev_log_likelihood = -np.inf
    num_states = hmm.num_states

    for em_round in range(NUM_EM_ROUNDS):
        client_statistics = []
        total_log_likelihood = 0.0
        for i, client_score_seqs in enumerate(client_score_sequences):
            if not client_score_seqs:
                continue
            agg_pi_update = np.zeros(num_states)
            agg_A_transitions = np.zeros((num_states, num_states))
            agg_digests = [TDigest() for _ in range(num_states)]
            client_ll = 0.0
            score_seqs = client_score_sequences[i] if client_score_sequences is not None else [None] * len(client_score_seqs)
            for seq_idx, seq_scores in enumerate(client_score_seqs):
                if len(seq_scores) <= 1:
                    continue
                seq_scores = score_seqs[seq_idx] if seq_idx < len(score_seqs) else None
                st, digests, ll = run_local_e_step(
                    seq_scores, hmm.pi, local_A_matrices[i],
                    hmm.get_emission_likelihood
                )
                agg_pi_update += st['pi_update']
                agg_A_transitions += st['A_transitions']
                for m in range(num_states):
                    agg_digests[m] += digests[m]
                client_ll += ll
            if np.sum(agg_pi_update) == 0:
                continue
            client_statistics.append({
                'client_index': i,
                'pi_update': agg_pi_update,
                'A_transitions': agg_A_transitions
            })
            all_client_per_state_digests[i] = agg_digests
            total_log_likelihood += client_ll

        if not client_statistics:
            break

        score_only_digests = [[] for _ in range(num_clients)]
        if client_score_sequences is not None:
            for i, client_score_seqs in enumerate(client_score_sequences):
                if not client_score_seqs:
                    continue
                agg_score_digests = [TDigest() for _ in range(num_states)]
                for seq_scores in client_score_seqs:
                    if len(seq_scores) <= 1:
                        continue
                    _, p_digests, _ = run_local_e_step(
                        seq_scores, hmm.pi, local_A_matrices[i],
                        hmm.get_emission_likelihood
                    )
                    for m in range(num_states):
                        agg_score_digests[m] += p_digests[m]
                score_only_digests[i] = agg_score_digests
        else:
            score_only_digests = all_client_per_state_digests

        hmm.global_m_step(client_statistics, score_only_digests)
        for st in client_statistics:
            ci = st['client_index']
            local_A_matrices[ci] = run_local_m_step(st['A_transitions'], hmm.zeta)
        if em_round > 0 and abs(total_log_likelihood - prev_log_likelihood) < CONVERGENCE_THRESHOLD:
            break
        prev_log_likelihood = total_log_likelihood

    if client_score_sequences is not None:
        for i, client_score_seqs in enumerate(client_score_sequences):
            if not client_score_seqs:
                continue
            score_seqs = client_score_sequences[i]
            agg_digests = [TDigest() for _ in range(num_states)]
            for seq_idx, seq_scores in enumerate(client_score_seqs):
                if len(seq_scores) <= 1:
                    continue
                seq_scores = score_seqs[seq_idx] if seq_idx < len(score_seqs) else None
                _, digests, _ = run_local_e_step(
                    seq_scores, hmm.pi, local_A_matrices[i],
                    hmm.get_emission_likelihood
                )
                for m in range(num_states):
                    agg_digests[m] += digests[m]
            all_client_per_state_digests[i] = agg_digests

    return hmm, local_A_matrices, all_client_per_state_digests


########################################
# SECTION 5: CALIBRATION AND EVALUATION
########################################

def compute_calibration_scores(model, units, seq_len=SEQ_LEN,
                               horizon=PREDICTION_HORIZON):
    model.eval()
    unit_scores = []
    all_scores_flat = []
    unit_predictions = []

    with torch.no_grad():
        for unit in units:
            signals = unit['signals']
            T = unit['T'] # Lifetime of a unit
            preds = []
            for t in range(0, T - seq_len - horizon + 1):
                x = torch.tensor(signals[t:t + seq_len], dtype=torch.float32).unsqueeze(0)
                out = model(x)
                pred_lo = out[0, 0].item()
                pred_med = out[0, 1].item()
                pred_hi = out[0, 2].item()
                y_true = signals[t + seq_len - 1 + horizon, 0]
                score = max(pred_lo - y_true, y_true - pred_hi)
                preds.append({
                    'pred_lower': pred_lo, 'pred_median': pred_med,
                    'pred_upper': pred_hi, 'label': y_true,
                    'score': score, 'time_idx': t + seq_len - 1,
                })
            if not preds:
                continue
            scores = np.array([p['score'] for p in preds])
            unit_scores.append(scores)
            all_scores_flat.extend(scores.tolist())
            unit_predictions.append(preds)

    return unit_scores, all_scores_flat, unit_predictions


def evaluate_conformal_methods(model, fleet_cal_data, test_units, num_fleets,
                               M_fit, a_diag=0.8, zeta_val=10.0, random_seed=0):
    # Compute scores per fleet
    client_score_sequences = [[] for _ in range(num_fleets)]
    all_cal_scores = []

    for k in range(num_fleets):
        if not fleet_cal_data[k]:
            continue
        for unit in fleet_cal_data[k]:
            u_scores, s_flat, _ = compute_calibration_scores(model, [unit])
            if u_scores and len(u_scores[0]) > 1:
                client_score_sequences[k].append(u_scores[0])
                all_cal_scores.extend(s_flat)

    N_cal = len(all_cal_scores)
    if N_cal == 0:
        return {'No CP': {}, 'Lu et al': {}, 'Proposed': {}}

    # Lu et al. FCP: static quantile (uniform over all calibration scores)
    target_rank = np.ceil((1 - ALPHA) * (N_cal + 1))
    p_level = target_rank / N_cal
    q_static = np.quantile(all_cal_scores, min(p_level, 1.0), method='higher') if p_level <= 1.0 else np.inf

    # Train HMM in a federated way
    hmm, final_A_matrices, all_client_per_state_digests = run_federated_hmm_training(
        NUM_HMM_STATES=M_fit, NUM_EM_ROUNDS=20, CONVERGENCE_THRESHOLD=1e-9,
        client_score_sequences=client_score_sequences,  
        all_cal_scores=all_cal_scores,
        a_diag=a_diag, zeta_val=zeta_val, random_seed=random_seed,
    )

    # Fleet-level and global digests for likelihood ratio weights
    client_overall_digests = [TDigest() for _ in range(num_fleets)]
    global_cal_digest = TDigest()
    for k, seq_list in enumerate(client_score_sequences):
        for seq in seq_list:
            for s in seq:
                client_overall_digests[k].update(s)
                global_cal_digest.update(s)
    client_pdfs = [hmm._build_pdf_from_digest(d) for d in client_overall_digests]
    global_pdf = hmm._build_pdf_from_digest(global_cal_digest)
    avg_A_matrix = calculate_global_average_A(final_A_matrices)

    # Evaluate test units
    records = []

    for test_unit in test_units:
        _, _, unit_preds = compute_calibration_scores(model, [test_unit])
        if not unit_preds or not unit_preds[0]:
            continue
        preds_list = unit_preds[0]

        # Evaluate at multiple time steps (every 5th step after context window)
        eval_indices = list(range(Tcw + 1, len(preds_list), 5))
        if len(preds_list) - 1 not in eval_indices:
            eval_indices.append(len(preds_list) - 1)

        for eval_idx in eval_indices:
            if eval_idx < Tcw + 1:
                continue
            p = preds_list[eval_idx]
            pl, pu, y_true = p['pred_lower'], p['pred_upper'], p['label']

            # True degradation stage at the target time
            target_t = p['time_idx'] + PREDICTION_HORIZON
            if target_t < len(test_unit['states']):
                true_stage = int(test_unit['states'][target_t])
            else:
                true_stage = -1

            # Context window for HMM state inference
            ctx_start = max(0, eval_idx - Tcw)
            context_scores = [preds_list[i]['score'] for i in range(ctx_start, eval_idx)]

            # Proposed method: adaptive quantile via HMM-weighted scores
            rho_m = predict_state_weights(context_scores, hmm.pi, avg_A_matrix,
                                          hmm.get_emission_likelihood)
            omega_k = get_likelihood_ratio_weights(context_scores, client_pdfs, global_pdf)
            q_adaptive = get_adaptive_quantile(rho_m, omega_k,
                                               all_client_per_state_digests, alpha=ALPHA)

            records.append({
                'y_true': y_true,
                'raw_lo': pl, 'raw_hi': pu,
                'static_lo': pl - q_static, 'static_hi': pu + q_static,
                'adaptive_lo': pl - q_adaptive, 'adaptive_hi': pu + q_adaptive,
                'true_stage': true_stage,
            })

    if not records:
        return {'No CP': {}, 'Lu et al': {}, 'Proposed': {}}

    def metrics_from_records(recs, lo_key, hi_key):
        y = np.array([r['y_true'] for r in recs])
        lo = np.array([r[lo_key] for r in recs])
        hi = np.array([r[hi_key] for r in recs])
        cov = float(np.mean((y >= lo) & (y <= hi))) * 100
        wid = float(np.mean(hi - lo))
        return cov, wid

    # Prediction intervals from different methods
    results = {}
    method_keys = {
        'No CP': ('raw_lo', 'raw_hi'),
        'Lu et al': ('static_lo', 'static_hi'),
        'Proposed': ('adaptive_lo', 'adaptive_hi'),
    } 
    
    for name, (lo_k, hi_k) in method_keys.items():
        cov_all, wid_all = metrics_from_records(records, lo_k, hi_k)
        results[name] = {
            'coverage': cov_all,
            'width': wid_all,
        }
        
        # Compute coverage and width separately for each true latent stage (per stage)
        stages_present = sorted(set(r['true_stage'] for r in records if r['true_stage'] >= 0))
        for stage in stages_present:
            stage_recs = [r for r in records if r['true_stage'] == stage]
            if len(stage_recs) >= 5:
                c_s, w_s = metrics_from_records(stage_recs, lo_k, hi_k)
                results[name][f'coverage_stage{stage}'] = c_s
                results[name][f'width_stage{stage}'] = w_s
                results[name][f'n_stage{stage}'] = len(stage_recs)

    return results


########################################
# SECTION 6: EXPERIMENT RUNNER
########################################

def prepare_experiment(dgp_params, use_federated=True, random_seed=42,
                       cal_ratio=0.3, num_test_units=50,
                       generate_diagnostics=False):
    """
    Generate data + train model. No HMM is included. 
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    dgp_params_clean = copy.deepcopy(dgp_params)
    test_reverse_prob = dgp_params_clean.pop("test_reverse_prob", 0.0)

    # Train/calibration DGP
    dgp = SyntheticDGP(**dgp_params_clean, random_state=random_seed)
    fleet_data = dgp.generate_dataset()

    # Default test fleet
    test_units = dgp.generate_test_fleet(num_units=num_test_units)

    # Experiment B: only test fleet violates left-to-right
    if test_reverse_prob > 0:
        test_A = np.mean(dgp.fleet_A, axis=0).copy()

        for m in range(1, dgp.M_true - 1):
            p_back = min(test_reverse_prob, test_A[m, m + 1] * 0.5)

            test_A[m, m - 1] += p_back
            test_A[m, m + 1] -= p_back

            test_A[m] = test_A[m] / test_A[m].sum()

        test_units = dgp.generate_test_fleet(
            num_units=num_test_units,
            fleet_transition=test_A
        )

    if generate_diagnostics:
        plot_diagnostics(dgp, fleet_data, tag=f"seed{random_seed}")

    num_fleets = dgp.K
    fleet_train = [[] for _ in range(num_fleets)]
    fleet_cal = [[] for _ in range(num_fleets)]
    for k in range(num_fleets):
        if len(fleet_data[k]) < 3:
            fleet_train[k] = fleet_data[k]
            continue
        tr, ca = train_test_split(fleet_data[k], test_size=cal_ratio, random_state=random_seed)
        fleet_train[k] = tr
        fleet_cal[k] = ca

    model = QuantileLSTM(input_size=N_SENSORS, hidden_size=32, num_layers=1)
    if use_federated:
        model = train_model_federated(model, fleet_train, rounds=10, epochs_local=5, lr=0.003)
    else:
        all_train = [u for fleet in fleet_train for u in fleet]
        model = train_model_centralized(model, all_train, epochs=30, lr=0.003)
    model.eval()

    return model, fleet_cal, test_units, num_fleets


def run_single_experiment(dgp_params, fit_M,
                          use_federated=True, random_seed=42,
                          a_diag=0.8, zeta_val=10.0,
                          cal_ratio=0.3, num_test_units=50,
                          generate_diagnostics=False,
                          precomputed=None):
    """
    Run one experiment. 
    """
    
    if precomputed is not None:
        model, fleet_cal, test_units, num_fleets = precomputed
    else:
        model, fleet_cal, test_units, num_fleets = prepare_experiment(
            dgp_params=dgp_params, use_federated=use_federated,
            random_seed=random_seed, cal_ratio=cal_ratio,
            num_test_units=num_test_units,
            generate_diagnostics=generate_diagnostics,
        )

    results = evaluate_conformal_methods(
        model=model, fleet_cal_data=fleet_cal, test_units=test_units,
        num_fleets=num_fleets, M_fit=fit_M,
        a_diag=a_diag, zeta_val=zeta_val, random_seed=random_seed,
    )
    
    return results



def run_experiment_with_replications(dgp_params, fit_M, n_reps=10,
                                     use_federated=True,
                                     a_diag=0.8, zeta_val=10.0,
                                     cal_ratio=0.3, num_test_units=50):
    all_results = []
    for rep in range(n_reps):
        seed = rep * 1000 + 42
        res = run_single_experiment(
            dgp_params=dgp_params, fit_M=fit_M,
            use_federated=use_federated, random_seed=seed,
            a_diag=a_diag, zeta_val=zeta_val,
            cal_ratio=cal_ratio, num_test_units=num_test_units,
            generate_diagnostics=(rep == 0),  # diagnostics for first rep only
        )
        all_results.append(res)

    agg = {} # To save aggregated results
    for method in ['No CP', 'Lu et al', 'Proposed']:
        covs = [r[method]['coverage'] for r in all_results if r[method]]
        wids = [r[method]['width'] for r in all_results if r[method]]
        if covs:
            d = {
                'coverage_mean': np.mean(covs),
                'coverage_std': np.std(covs),
                'width_mean': np.mean(wids),
                'width_std': np.std(wids),
            }
            # Aggregate per-stage metrics
            for stage in range(10):  
                cov_key = f'coverage_stage{stage}'
                wid_key = f'width_stage{stage}'
                stage_covs = [r[method][cov_key] for r in all_results
                              if r[method] and cov_key in r[method]]
                stage_wids = [r[method][wid_key] for r in all_results
                              if r[method] and wid_key in r[method]]
                if stage_covs:
                    d[f'{cov_key}_mean'] = np.mean(stage_covs)
                    d[f'{wid_key}_mean'] = np.mean(stage_wids)
            agg[method] = d
        else:
            agg[method] = {'coverage_mean': np.nan, 'coverage_std': np.nan,
                           'width_mean': np.nan, 'width_std': np.nan}
    return agg


########################################
# SECTION 7: MAIN
########################################

def main():
    N_REPS = 10
    NUM_TEST = 20
    summary_rows = []

    base_dgp = dict(
        num_true_states=4,
        num_fleets=5,
        units_per_fleet=[20, 20, 20, 20, 20],
        trajectory_range=(80, 150),
        dirichlet_concentration=50.0,
        fleet_emission_shift=0.0,
        reverse_prob=0.0,
        base_transition_stay=0.96,
        signal_momentum=0.5,
    )

    print("Generating visualization plots for base data generating process...")
    dgp_check = SyntheticDGP(**base_dgp, random_state=42)
    fd_check = dgp_check.generate_dataset()
    plot_diagnostics(dgp_check, fd_check, tag="base_check")
    print()

    print("=" * 70)
    print("EXPERIMENT A: Sensitivity to number of HMM states M")
    print("=" * 70)

    for use_fed in [True, False]:
        label = "Federated" if use_fed else "Centralized"
        
        #modify 0 to N_REPS
        for rep in range(N_REPS):
            seed = rep * 1000 + 42
            precomp = prepare_experiment(
                dgp_params=base_dgp, use_federated=use_fed,
                random_seed=seed, num_test_units=NUM_TEST,
                generate_diagnostics=(rep == 0 and use_fed),
            )
            for M_fit in [2, 3, 4, 5, 7]:
                res = run_single_experiment(
                    dgp_params=base_dgp, fit_M=M_fit,
                    use_federated=use_fed, random_seed=seed,
                    num_test_units=NUM_TEST, precomputed=precomp,
                )
                
                for method, vals in res.items():
                    if vals:
                        summary_rows.append({
                            'Experiment': 'A_M_sensitivity', 'M_true': 4, 'M_fit': M_fit,
                            'Training': label, 'reverse_prob': 0.0,
                            'emission_shift': 0.0, 'dirichlet_conc': 50.0,
                            'CP_Method': method, 'rep': rep,
                            'Coverage': vals['coverage'], 'Width': vals['width'],
                        })
            print(f"  {label} rep {rep+1}/{N_REPS} done")
        print()

    print("\n" + "=" * 70)
    print("EXPERIMENT B: Violation of left-to-right structure")
    print("=" * 70)

    for p_rev in [0.0, 0.05, 0.1, 0.2]:
        dgp_B = copy.deepcopy(base_dgp)
        dgp_B['reverse_prob'] = p_rev
        for use_fed in [True, False]:
            label = "Federated" if use_fed else "Centralized"
            print(f"  reverse_prob={p_rev}, {label} ...")
            for rep in range(N_REPS):
                seed = rep * 1000 + 42
                res = run_single_experiment(
                    dgp_params=dgp_B, fit_M=4, use_federated=use_fed,
                    random_seed=seed, num_test_units=NUM_TEST,
                )
                for method, vals in res.items():
                    if vals:
                        summary_rows.append({
                            'Experiment': 'B_left_to_right', 'M_true': 4, 'M_fit': 4,
                            'Training': label, 'reverse_prob': p_rev,
                            'emission_shift': 0.0, 'dirichlet_conc': 50.0,
                            'CP_Method': method, 'rep': rep,
                            'Coverage': vals['coverage'], 'Width': vals['width'],
                        })

    print("\n" + "=" * 70)
    print("EXPERIMENT C: Violation of shared-emission assumption")
    print("=" * 70)

    for delta in [0.0, 0.1, 0.3, 0.5, 1.0]:
        dgp_C = copy.deepcopy(base_dgp)
        dgp_C['fleet_emission_shift'] = delta
        for use_fed in [True, False]:
            label = "Federated" if use_fed else "Centralized"
            print(f"  emission_shift={delta}, {label} ...")

            #modify 0 to N_REPS
            for rep in range(0):
                seed = rep * 1000 + 42
                res = run_single_experiment(
                    dgp_params=dgp_C, fit_M=4, use_federated=use_fed,
                    random_seed=seed, num_test_units=NUM_TEST,
                )
                for method, vals in res.items():
                    if vals:
                        summary_rows.append({
                            'Experiment': 'C_shared_emission', 'M_true': 4, 'M_fit': 4,
                            'Training': label, 'reverse_prob': 0.0,
                            'emission_shift': delta, 'dirichlet_conc': 50.0,
                            'CP_Method': method, 'rep': rep,
                            'Coverage': vals['coverage'], 'Width': vals['width'],
                        })

    print("\n" + "=" * 70)
    print("EXPERIMENT D: Fleet heterogeneity in transition dynamics")
    print("=" * 70)

    for conc in [5.0, 15.0, 50.0, 150.0, 500.0]:
        dgp_D = copy.deepcopy(base_dgp)
        dgp_D['dirichlet_concentration'] = conc
        for use_fed in [True, False]:
            label = "Federated" if use_fed else "Centralized"
            print(f"  dirichlet_conc={conc}, {label} ...")

            #modify 0 to N_REPS
            for rep in range(N_REPS):
                seed = rep * 1000 + 42
                res = run_single_experiment(
                    dgp_params=dgp_D, fit_M=4, use_federated=use_fed,
                    random_seed=seed, num_test_units=NUM_TEST,
                )
                for method, vals in res.items():
                    if vals:
                        summary_rows.append({
                            'Experiment': 'D_fleet_heterogeneity', 'M_true': 4, 'M_fit': 4,
                            'Training': label, 'reverse_prob': 0.0,
                            'emission_shift': 0.0, 'dirichlet_conc': conc,
                            'CP_Method': method, 'rep': rep,
                            'Coverage': vals['coverage'], 'Width': vals['width'],
                        })

    
    df = pd.DataFrame(summary_rows)
    raw_path = r"C:\Users\dogha\Desktop\HMM_simulation\Simulation_Results\Outputs\simulation_robustness_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw per-rep results saved to {raw_path}")


    agg_df = _aggregate_results(df)
    agg_path = r"C:\Users\dogha\Desktop\HMM_simulation\Simulation_Results\Outputs\simulation_robustness_summary.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"Aggregated summary saved to {agg_path}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLES")
    print("=" * 70)
    pivot_cols = {
        'A_M_sensitivity': 'M_fit',
        'B_left_to_right': 'reverse_prob',
        'C_shared_emission': 'emission_shift',
        'D_fleet_heterogeneity': 'dirichlet_conc',
    }
    for exp_name, col in pivot_cols.items():
        sub = agg_df[agg_df['Experiment'] == exp_name]
        if sub.empty:
            continue
        print(f"\n--- {exp_name} ---")
        for training_type in ['Federated', 'Centralized']:
            sub_t = sub[sub['Training'] == training_type]
            if sub_t.empty:
                continue
            print(f"\n  {training_type}:")
            vals = sorted(sub_t[col].unique())
            header = f"  {'Method':<12}"
            for v in vals:
                header += f"  {col}={v}"
            print(header)
            for method in ['No CP', 'Lu et al', 'Proposed']:
                line = f"  {method:<12}"
                for v in vals:
                    row = sub_t[(sub_t[col] == v) & (sub_t['CP_Method'] == method)]
                    if not row.empty:
                        c = row.iloc[0]['Coverage_Mean']
                        w = row.iloc[0]['Width_Mean']
                        line += f"  {c:5.1f}/{w:.4f}"
                    else:
                        line += f"  {'N/A':>12}"
                print(line)
    return df


def _aggregate_results(df):
    group_cols = ['Experiment', 'M_true', 'M_fit', 'Training',
                  'reverse_prob', 'emission_shift', 'dirichlet_conc', 'CP_Method']
    agg = df.groupby(group_cols).agg(
        Coverage_Mean=('Coverage', 'mean'),
        Coverage_Std=('Coverage', 'std'),
        Width_Mean=('Width', 'mean'),
        Width_Std=('Width', 'std'),
        N_reps=('rep', 'count'),
    ).reset_index()
    return agg

if __name__ == "__main__":
    results_df = main()