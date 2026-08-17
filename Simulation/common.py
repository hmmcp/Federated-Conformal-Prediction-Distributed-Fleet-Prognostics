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

script_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(script_dir, "Outputs")
diagnostics_dir = os.path.join(script_dir, "Diagnostics")

os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(diagnostics_dir, exist_ok=True)
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
    # use your own path for diagnostics visualization
    fig_dir = diagnostics_dir

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
    axes[-1].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/sample_units_sensor0_{tag}.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(dgp.K, 1, figsize=(14, 3 * dgp.K), sharex=False)
    if dgp.K == 1:
        axes = [axes]
    for k in range(dgp.K):
        ax = axes[k]
        unit = fleet_data[k][0]
        sig = unit['signals'][:, 0]
        st = unit['states']
        for m in range(dgp.M_true):
            mask = (st == m)
            ax.scatter(np.where(mask)[0], sig[mask], s=12, label=f"state {m}")
        ax.plot(sig, alpha=0.25, color="gray")
        ax.set_title(f"Fleet {k+1}: one unit with latent states (sensor 0)", fontsize=11)
        ax.set_ylabel("signal")
        ax.legend(fontsize=8, ncol=dgp.M_true)
    axes[-1].set_xlabel("time")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/stage_colored_sensor0_{tag}.png", dpi=150)
    plt.close()

    occ = np.zeros((dgp.K, dgp.M_true))
    for k in range(dgp.K):
        counts = np.zeros(dgp.M_true)
        total = 0
        for unit in fleet_data[k]:
            for m in range(dgp.M_true):
                counts[m] += np.sum(unit['states'] == m)
            total += len(unit['states'])
        occ[k] = counts / max(total, 1)

    plt.figure(figsize=(8, 4))
    x = np.arange(dgp.M_true)
    bottom = np.zeros(dgp.M_true)
    for k in range(dgp.K):
        plt.bar(x, occ[k], bottom=bottom, alpha=0.6, label=f"fleet {k+1}")
        bottom += occ[k]
    plt.xticks(x, [f"state {m}" for m in range(dgp.M_true)])
    plt.ylabel("stacked occupancy")
    plt.title("State occupancy by fleet")
    plt.legend(fontsize=8, ncol=min(dgp.K, 3))
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/stage_occupancy_by_fleet_{tag}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    for m in range(dgp.M_true):
        plt.plot(np.arange(dgp.n_sensors), dgp.emission_means[m], marker='o', label=f"state {m}")
    plt.xticks(np.arange(dgp.n_sensors), [f"s{i}" for i in range(dgp.n_sensors)])
    plt.ylabel("mean")
    plt.title("Emission means across sensors by state")
    plt.legend(fontsize=8, ncol=min(dgp.M_true, 4))
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/stage_means_sensor0_{tag}.png", dpi=150)
    plt.close()

    print("  Stage occupancy (fleet-averaged):", np.round(np.mean(occ, axis=0), 3))
    print("  Emission means (sensor 0):", np.round(dgp.emission_means[:, 0], 3))
    print("  Emission stds (sensor 0): ", np.round(dgp.emission_stds[:, 0], 3))
    for k in range(dgp.K):
        print(f"  Fleet {k+1} A diagonal:", np.round(np.diag(dgp.fleet_A[k]), 4))


########################################
# SECTION 2: DATASET PREPARATION
########################################

class FleetSequenceDataset(Dataset):
    def __init__(self, units, seq_len=20):
        self.samples = []
        self.seq_len = seq_len
        for unit in units:
            X = unit['signals']
            T = unit['T']
            n = X.shape[0]
            if n < seq_len + PREDICTION_HORIZON:
                continue
            for t in range(seq_len - 1, n - PREDICTION_HORIZON):
                x_seq = X[t - seq_len + 1:t + 1]
                y = X[t + PREDICTION_HORIZON, 0]
                rul = T - (t + PREDICTION_HORIZON + 1)
                self.samples.append((x_seq.astype(np.float32), np.float32(y), rul))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y, rul = self.samples[idx]
        return torch.tensor(x_seq), torch.tensor(y), torch.tensor(rul, dtype=torch.float32)


def split_train_calibration(fleet_units, train_ratio=0.7, random_state=42):
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(fleet_units))
    rng.shuffle(idx)
    n_train = int(np.floor(train_ratio * len(fleet_units)))
    train_idx = idx[:n_train]
    cal_idx = idx[n_train:]
    train_units = [fleet_units[i] for i in train_idx]
    cal_units = [fleet_units[i] for i in cal_idx]
    return train_units, cal_units


########################################
# SECTION 3: QUANTILE REGRESSION MODEL
########################################

class QuantileLSTM(nn.Module):
    def __init__(self, input_size=N_SENSORS, hidden_size=64, num_layers=1, quantiles=None):
        super().__init__()
        self.quantiles = quantiles or TRAIN_QUANTILES
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, len(self.quantiles))

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        q = self.fc(h)
        return q


def pinball_loss(preds, target, quantiles):
    target = target.view(-1, 1)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i:i+1]
        losses.append(torch.maximum((q - 1) * errors, q * errors).mean())
    return sum(losses) / len(losses)


def train_qr_model(train_units, device='cpu', epochs=20, lr=1e-3, verbose=False):
    ds = FleetSequenceDataset(train_units, seq_len=SEQ_LEN)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = QuantileLSTM().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        losses = []
        for xb, yb, _ in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = pinball_loss(preds, yb, model.quantiles)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if verbose and ((ep + 1) % 5 == 0 or ep == 0):
            print(f"    QR epoch {ep+1}/{epochs}, loss={np.mean(losses):.4f}")
    return model


def federated_train_qr(client_train_units, device='cpu', rounds=10, local_epochs=3, lr=1e-3, verbose=False):
    global_model = QuantileLSTM().to(device)
    global_sd = global_model.state_dict()

    for r in range(rounds):
        local_sds = []
        local_ns = []
        for k, units in enumerate(client_train_units):
            ds = FleetSequenceDataset(units, seq_len=SEQ_LEN)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
            local_model = QuantileLSTM().to(device)
            local_model.load_state_dict(global_sd)
            opt = optim.Adam(local_model.parameters(), lr=lr)
            local_model.train()
            for ep in range(local_epochs):
                for xb, yb, _ in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    preds = local_model(xb)
                    loss = pinball_loss(preds, yb, local_model.quantiles)
                    loss.backward()
                    opt.step()
            local_sds.append({k0: v.detach().cpu().clone() for k0, v in local_model.state_dict().items()})
            local_ns.append(len(ds))
        if not local_sds:
            return global_model
        total_n = float(sum(local_ns))
        new_sd = {}
        for key in global_sd.keys():
            acc = sum(sd[key] * (n / total_n) for sd, n in zip(local_sds, local_ns))
            new_sd[key] = acc
        global_sd = new_sd
        global_model.load_state_dict(global_sd)
        if verbose:
            print(f"    Federated round {r+1}/{rounds} done")
    return global_model


########################################
# SECTION 4: CONFORMAL SCORES & PROXIES
########################################

def collect_unit_predictions(model, unit, device='cpu'):
    model.eval()
    X = unit['signals']
    T = unit['T']
    preds = []
    with torch.no_grad():
        for t in range(SEQ_LEN - 1, len(X) - PREDICTION_HORIZON):
            x_seq = X[t - SEQ_LEN + 1:t + 1].astype(np.float32)
            xb = torch.tensor(x_seq).unsqueeze(0).to(device)
            q = model(xb).cpu().numpy().reshape(-1)
            y_true = X[t + PREDICTION_HORIZON, 0]
            rul = T - (t + PREDICTION_HORIZON + 1)
            preds.append({
                't': t,
                'q_lo': float(q[0]),
                'q_med': float(q[1]),
                'q_hi': float(q[2]),
                'y_true': float(y_true),
                'rul': int(rul),
            })
    return preds


def compute_scores_and_proxies(pred_list):
    scores = []
    proxies = []
    for i, d in enumerate(pred_list):
        s = max(d['q_lo'] - d['y_true'], d['y_true'] - d['q_hi'])
        scores.append(float(s))
        if i == 0:
            proxy = d['q_hi'] - d['q_lo']
        else:
            proxy = abs(d['q_med'] - (pred_list[i-1]['q_med']))
        proxies.append(float(proxy))
    return np.array(scores), np.array(proxies)


########################################
# SECTION 5: FEDERATED HMM
########################################

class SimpleFederatedHMM:
    def __init__(self, num_states=4, a_diag=0.85, zeta_val=10.0, random_state=42):
        self.M = num_states
        self.rng = np.random.default_rng(random_state)
        self.pi = np.ones(self.M) / self.M
        self.A = np.zeros((self.M, self.M))
        for m in range(self.M - 1):
            self.A[m, m] = a_diag
            self.A[m, m + 1] = 1 - a_diag
        self.A[self.M - 1, self.M - 1] = 1.0
        self.zeta = np.full((self.M - 1, 2), zeta_val)
        self.global_state_digests = [TDigest() for _ in range(self.M)]

    def _emission_lik(self, x, m):
        if self.global_state_digests[m].n < 5:
            return 1e-6
        q25 = self.global_state_digests[m].percentile(25)
        q50 = self.global_state_digests[m].percentile(50)
        q75 = self.global_state_digests[m].percentile(75)
        scale = max(q75 - q25, 1e-3)
        return np.exp(-abs(x - q50) / scale) / (2 * scale)

    def forward_filter(self, obs, A_local=None):
        A_use = self.A if A_local is None else A_local
        Tn = len(obs)
        alpha = np.zeros((Tn, self.M))
        c = np.zeros(Tn)
        b0 = np.array([self._emission_lik(obs[0], m) for m in range(self.M)])
        alpha[0] = self.pi * b0
        c[0] = alpha[0].sum() + 1e-12
        alpha[0] /= c[0]
        for t in range(1, Tn):
            bt = np.array([self._emission_lik(obs[t], m) for m in range(self.M)])
            alpha[t] = alpha[t - 1] @ A_use * bt
            c[t] = alpha[t].sum() + 1e-12
            alpha[t] /= c[t]
        return alpha, c, A_use

    def backward_smooth(self, obs, alpha, c, A_local=None):
        A_use = self.A if A_local is None else A_local
        Tn = len(obs)
        beta = np.zeros((Tn, self.M))
        beta[-1] = 1.0
        for t in range(Tn - 2, -1, -1):
            bt1 = np.array([self._emission_lik(obs[t + 1], m) for m in range(self.M)])
            beta[t] = (A_use * bt1).dot(beta[t + 1]) / c[t + 1]
        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
        return gamma


def initialize_hmm_from_all_proxies(all_proxy_sequences, num_states=4, random_state=42):
    hmm = SimpleFederatedHMM(num_states=num_states, random_state=random_state)
    flat = np.concatenate([np.asarray(seq).reshape(-1) for seq in all_proxy_sequences if len(seq) > 0])
    if flat.size == 0:
        return hmm
    qs = np.quantile(flat, np.linspace(0, 1, num_states + 1))
    for x in flat:
        idx = np.searchsorted(qs[1:-1], x)
        hmm.global_state_digests[idx].update(float(x))
    return hmm


def local_e_step(hmm, proxy_seq, score_seq=None, A_local=None):
    proxy_seq = np.asarray(proxy_seq).reshape(-1)
    if len(proxy_seq) == 0:
        return None
    alpha, c, A_use = hmm.forward_filter(proxy_seq, A_local=A_local)
    gamma = hmm.backward_smooth(proxy_seq, alpha, c, A_local=A_use)
    xi_sum = np.zeros((hmm.M, hmm.M))
    for t in range(len(proxy_seq) - 1):
        bnext = np.array([hmm._emission_lik(proxy_seq[t + 1], m) for m in range(hmm.M)])
        numer = (alpha[t][:, None] * A_use) * (bnext[None, :] * (gamma[t + 1] / np.maximum(alpha[t + 1], 1e-12))[None, :])
        denom = np.maximum(numer.sum(), 1e-12)
        xi_sum += numer / denom
    state_digests = [TDigest() for _ in range(hmm.M)]
    vals = proxy_seq if score_seq is None else np.asarray(score_seq).reshape(-1)
    L = min(len(vals), len(proxy_seq))
    for t in range(L):
        for m in range(hmm.M):
            w = gamma[t, m]
            if w > 1e-8:
                state_digests[m].update(float(vals[t]), w)
    return {
        'gamma0': gamma[0],
        'xi_sum': xi_sum,
        'state_digests': state_digests,
        'loglik': float(np.sum(np.log(c + 1e-12))),
        'gamma': gamma,
    }


def local_m_step(A_counts, zeta):
    M = A_counts.shape[0]
    A_new = np.zeros((M, M))
    for m in range(M - 1):
        N_mm = A_counts[m, m]
        N_mn = A_counts[m, m + 1]
        z_mm, z_mn = zeta[m]
        denom = (N_mm + N_mn) + (z_mm + z_mn) - 2
        if denom <= 0:
            A_new[m, m] = 0.5
            A_new[m, m + 1] = 0.5
        else:
            A_new[m, m] = (N_mm + z_mm - 1) / denom
            A_new[m, m + 1] = (N_mn + z_mn - 1) / denom
    A_new[M - 1, M - 1] = 1.0
    return A_new


def federated_gem_hmm(client_proxy_sequences, client_score_sequences=None,
                      num_states=4, num_rounds=20, tol=1e-4,
                      a_diag=0.85, zeta_val=10.0, random_state=42):
    all_proxy_flat = [seq for client_seqs in client_proxy_sequences for seq in client_seqs]
    hmm = initialize_hmm_from_all_proxies(all_proxy_flat, num_states=num_states, random_state=random_state)
    hmm.A = np.zeros((num_states, num_states))
    for m in range(num_states - 1):
        hmm.A[m, m] = a_diag
        hmm.A[m, m + 1] = 1 - a_diag
    hmm.A[num_states - 1, num_states - 1] = 1.0
    hmm.zeta = np.full((num_states - 1, 2), zeta_val)

    K = len(client_proxy_sequences)
    local_As = [hmm.A.copy() for _ in range(K)]
    prev_ll = -np.inf

    for r in range(num_rounds):
        pi_acc = np.zeros(hmm.M)
        count_clients = 0
        local_counts = [np.zeros((hmm.M, hmm.M)) for _ in range(K)]
        proxy_digests_global = [TDigest() for _ in range(hmm.M)]
        score_digests_by_client = [[TDigest() for _ in range(hmm.M)] for _ in range(K)]
        total_ll = 0.0

        for k in range(K):
            client_seqs = client_proxy_sequences[k]
            client_score_seqs = client_score_sequences[k] if client_score_sequences is not None else [None] * len(client_seqs)
            if len(client_seqs) == 0:
                continue
            gamma0_sum = np.zeros(hmm.M)
            xi_sum = np.zeros((hmm.M, hmm.M))
            valid = 0
            for seq_idx, proxy_seq in enumerate(client_seqs):
                if len(proxy_seq) < 2:
                    continue
                out_proxy = local_e_step(hmm, proxy_seq, score_seq=None, A_local=local_As[k])
                out_score = local_e_step(hmm, proxy_seq,
                                         score_seq=client_score_seqs[seq_idx] if seq_idx < len(client_score_seqs) else None,
                                         A_local=local_As[k])
                if out_proxy is None or out_score is None:
                    continue
                gamma0_sum += out_proxy['gamma0']
                xi_sum += out_proxy['xi_sum']
                total_ll += out_proxy['loglik']
                valid += 1
                for m in range(hmm.M):
                    proxy_digests_global[m] += out_proxy['state_digests'][m]
                    score_digests_by_client[k][m] += out_score['state_digests'][m]
            if valid > 0:
                pi_acc += gamma0_sum / valid
                local_counts[k] = xi_sum
                count_clients += 1

        if count_clients == 0:
            break
        hmm.pi = pi_acc / np.maximum(pi_acc.sum(), 1e-12)
        hmm.global_state_digests = proxy_digests_global
        for k in range(K):
            local_As[k] = local_m_step(local_counts[k], hmm.zeta)

        if abs(total_ll - prev_ll) < tol:
            break
        prev_ll = total_ll

    return hmm, local_As, score_digests_by_client


def predict_state_weights(proxy_context, hmm, A_avg):
    proxy_context = np.asarray(proxy_context).reshape(-1)
    if len(proxy_context) == 0:
        return hmm.pi.copy()
    alpha, _, _ = hmm.forward_filter(proxy_context, A_local=A_avg)
    return alpha[-1].copy()


def _pdf_from_digest(dig, bins=100):
    if dig.n < 2:
        xs = np.array([0.0, 1.0])
        ps = np.array([1.0])
        return xs, ps
    lo = dig.percentile(1)
    hi = dig.percentile(99)
    if hi <= lo:
        hi = lo + 1e-3
    xs = np.linspace(lo, hi, bins + 1)
    mids = 0.5 * (xs[:-1] + xs[1:])
    cdf_hi = np.array([dig.cdf(x) for x in xs[1:]])
    cdf_lo = np.array([dig.cdf(x) for x in xs[:-1]])
    probs = np.maximum(cdf_hi - cdf_lo, 1e-9)
    probs = probs / probs.sum()
    return xs, probs


def likelihood_ratio_weights(proxy_context, client_proxy_digests, global_proxy_digests):
    pdf_clients = [_pdf_from_digest(d) for d in client_proxy_digests]
    pdf_global = _pdf_from_digest(global_proxy_digests)

    def score_loglik(seq, pdf):
        xs, probs = pdf
        mids = 0.5 * (xs[:-1] + xs[1:])
        ll = 0.0
        for x in seq:
            j = np.clip(np.searchsorted(mids, x), 0, len(probs) - 1)
            ll += np.log(probs[j] + 1e-12)
        return ll

    ll_g = score_loglik(proxy_context, pdf_global)
    lrs = []
    for pdf in pdf_clients:
        ll_k = score_loglik(proxy_context, pdf)
        lrs.append(np.exp(ll_k - ll_g))
    lrs = np.array(lrs)
    if np.all(lrs == 0):
        return np.ones_like(lrs) / len(lrs)
    return lrs / lrs.sum()


def adaptive_quantile_from_digests(rho_m, omega_k, client_state_score_digests, alpha=ALPHA):
    final_dig = TDigest()
    K = len(client_state_score_digests)
    M = len(rho_m)
    for k in range(K):
        for m in range(M):
            dig = client_state_score_digests[k][m]
            if dig.n == 0:
                continue
            w = float(rho_m[m] * omega_k[k])
            if w <= 0:
                continue
            for c in dig.centroids_to_list():
                final_dig.update(c['m'], c['c'] * w)
    if final_dig.n == 0:
        return np.inf
    return final_dig.percentile(100 * (1 - alpha))


########################################
# SECTION 6: FULL PIPELINE
########################################

def prepare_experiment(dgp_params, use_federated=True, fit_M=4, random_seed=42,
                       num_test_units=20, generate_diagnostics=False):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dgp = SyntheticDGP(**dgp_params, random_state=random_seed)
    fleet_data = dgp.generate_dataset()
    if generate_diagnostics:
        plot_diagnostics(dgp, fleet_data, tag=f"seed{random_seed}")

    client_train_units = []
    client_cal_units = []
    for k in range(dgp.K):
        tr_u, cal_u = split_train_calibration(fleet_data[k], train_ratio=0.7, random_state=random_seed + k)
        client_train_units.append(tr_u)
        client_cal_units.append(cal_u)

    if use_federated:
        qr_model = federated_train_qr(client_train_units, device=device, rounds=10, local_epochs=3, lr=1e-3, verbose=False)
    else:
        all_train = [u for cu in client_train_units for u in cu]
        qr_model = train_qr_model(all_train, device=device, epochs=30, lr=1e-3, verbose=False)

    A_test = np.mean(np.stack(dgp.fleet_A, axis=0), axis=0).copy()
    if dgp.reverse_prob == 0:
        for m in range(dgp.M_true - 1):
            A_test[m, m] = max(A_test[m, m] - 0.05, 0.70)
            A_test[m, m + 1] = 1.0 - A_test[m, m]
        A_test[dgp.M_true - 1, dgp.M_true - 1] = 1.0

    test_units = dgp.generate_test_fleet(num_units=num_test_units, fleet_transition=A_test)

    return {
        'dgp': dgp,
        'fleet_data': fleet_data,
        'client_train_units': client_train_units,
        'client_cal_units': client_cal_units,
        'qr_model': qr_model,
        'test_units': test_units,
        'device': device,
        'fit_M': fit_M,
        'use_federated': use_federated,
        'random_seed': random_seed,
    }


def evaluate_conformal_methods(precomp):
    qr_model = precomp['qr_model']
    if qr_model is None:
        return None
    device = precomp['device']
    client_cal_units = precomp['client_cal_units']
    test_units = precomp['test_units']
    fit_M = precomp['fit_M']
    random_seed = precomp['random_seed']

    client_score_sequences = []
    client_proxy_sequences = []
    client_overall_proxy_digests = []
    all_scores_full = []
    for cal_units in client_cal_units:
        seq_scores_k = []
        seq_proxies_k = []
        dig_all_proxy = TDigest()
        for unit in cal_units:
            pred_list = collect_unit_predictions(qr_model, unit, device=device)
            if len(pred_list) == 0:
                continue
            scores, proxies = compute_scores_and_proxies(pred_list)
            seq_scores_k.append(scores)
            seq_proxies_k.append(proxies)
            for s in scores:
                all_scores_full.append(float(s))
            for p in proxies:
                dig_all_proxy.update(float(p))
        client_score_sequences.append(seq_scores_k)
        client_proxy_sequences.append(seq_proxies_k)
        client_overall_proxy_digests.append(dig_all_proxy)

    Ncal = len(all_scores_full)
    q_static = np.inf if Ncal == 0 else np.quantile(
        np.asarray(all_scores_full),
        min(np.ceil((1 - ALPHA) * (Ncal + len(client_cal_units))) / Ncal, 1.0),
        method='higher'
    )

    hmm, local_As, client_state_score_digests = federated_gem_hmm(
        client_proxy_sequences, client_score_sequences=client_score_sequences,
        num_states=fit_M, num_rounds=20, tol=1e-6,
        a_diag=0.8, zeta_val=5.0, random_state=random_seed
    )
    A_avg = np.mean(np.stack(local_As, axis=0), axis=0)
    global_proxy_digest = TDigest()
    for d in client_overall_proxy_digests:
        global_proxy_digest += d

    covered = {'No CP': [], 'Lu et al': [], 'Proposed': []}
    widths = {'No CP': [], 'Lu et al': [], 'Proposed': []}
    by_bin = {'No CP': [], 'Lu et al': [], 'Proposed': []}

    for unit in test_units:
        pred_list = collect_unit_predictions(qr_model, unit, device=device)
        if len(pred_list) < 2:
            continue
        scores_test, proxies_test = compute_scores_and_proxies(pred_list)
        d = pred_list[-1]
        y = d['y_true']
        rul = d['rul']
        lo0, hi0 = d['q_lo'], d['q_hi']

        lo_raw, hi_raw = lo0, hi0
        lo_static, hi_static = lo0 - q_static, hi0 + q_static

        context = proxies_test[max(0, len(proxies_test) - Tcw - 1):-1] if len(proxies_test) > 1 else np.array([])
        rho_m = predict_state_weights(context, hmm, A_avg)
        omega_k = likelihood_ratio_weights(context, client_overall_proxy_digests, global_proxy_digest)
        q_adapt = adaptive_quantile_from_digests(rho_m, omega_k, client_state_score_digests, alpha=ALPHA)
        lo_prop, hi_prop = lo0 - q_adapt, hi0 + q_adapt

        intervals = {
            'No CP': (lo_raw, hi_raw),
            'Lu et al': (lo_static, hi_static),
            'Proposed': (lo_prop, hi_prop),
        }

        for method, (lo, hi) in intervals.items():
            is_cov = float((y >= lo) and (y <= hi))
            wid = float(hi - lo)
            covered[method].append(is_cov)
            widths[method].append(wid)
            by_bin[method].append((rul, is_cov, wid))

    results = {}
    for method in ['No CP', 'Lu et al', 'Proposed']:
        if len(covered[method]) == 0:
            results[method] = None
            continue
        results[method] = {
            'coverage': float(np.mean(covered[method])),
            'width': float(np.mean(widths[method])),
            'width_sd': float(np.std(widths[method])),
            'by_bin': by_bin[method],
        }
    return results


def run_single_experiment(dgp_params, fit_M=4, use_federated=True, random_seed=42, num_test_units=20,
                          precomputed=None):
    precomp = precomputed if precomputed is not None else prepare_experiment(
        dgp_params=dgp_params,
        use_federated=use_federated,
        fit_M=fit_M,
        random_seed=random_seed,
        num_test_units=num_test_units,
        generate_diagnostics=False,
    )
    return evaluate_conformal_methods(precomp)


def summarize_by_bins(by_bin_records):
    bins = [(-np.inf, 50, 'lt50'), (50, 100, '50_100'), (100, np.inf, 'gt100')]
    out = {}
    arr = np.array(by_bin_records, dtype=object)
    if len(arr) == 0:
        for _, _, name in bins:
            out[name] = {'coverage_mean': np.nan, 'coverage_std': np.nan,
                         'width_mean': np.nan, 'width_std': np.nan}
        return out
    rul = arr[:, 0].astype(float)
    cov = arr[:, 1].astype(float)
    wid = arr[:, 2].astype(float)
    for lo, hi, name in bins:
        mask = (rul > lo) & (rul <= hi)
        if np.sum(mask) > 0:
            out[name] = {'coverage_mean': cov[mask].mean(), 'coverage_std': cov[mask].std(),
                         'width_mean': wid[mask].mean(), 'width_std': wid[mask].std()}
        else:
            out[name] = {'coverage_mean': np.nan, 'coverage_std': np.nan,
                         'width_mean': np.nan, 'width_std': np.nan}
    return out


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
