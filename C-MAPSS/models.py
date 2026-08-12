#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model definitions for the HMM-FCP experiment."""

import numpy as np
import torch
import torch.nn as nn
from tdigest import TDigest

ALPHA = 0.1
RUL_SCALE = 100
delta_Value = 0.01

class QuantileLSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, dropout=0.1):
        super(QuantileLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 3)  # [lower, median, upper]

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output

class FederatedHMM:
    def __init__(self, num_states, initial_scores, a_diag=0.8, zeta_val=10.0, random_state=None):
        rng = np.random.default_rng(random_state)
        self.num_states = num_states
        
        self.pi = rng.dirichlet(alpha=np.ones(num_states))
        
        self.emission_digests = [TDigest(delta=delta_Value) for _ in range(num_states)]
        if initial_scores:
            for i, score in enumerate(initial_scores):
                self.emission_digests[i % num_states].update(score)
        self.emission_pdfs = [self._build_pdf_from_digest(d) for d in self.emission_digests]
        
        self.A_init = np.zeros((num_states, num_states))
        for i in range(num_states - 1):
            self.A_init[i, i] = a_diag
            self.A_init[i, i+1] = 1.0 - a_diag
        self.A_init[num_states-1, num_states-1] = 1.0
        
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
        boundaries = [(means[i] + means[i+1]) / 2 for i in range(len(means) - 1)]
        return probs, boundaries

    def _get_emission_likelihood_from_pdfs(self, score, state_idx, emission_pdfs):
        probs, boundaries = emission_pdfs[state_idx]
        if not probs:
            return 1e-9
        j = np.searchsorted(boundaries, score)
        p = probs[j]
        return max(p, 1e-9)

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
        new_emission_digests = [TDigest(delta=delta_Value) for _ in range(self.num_states)]
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

