#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data processing, federated learning, and conformal prediction functions."""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from tdigest import TDigest
from pathlib import Path

from models import ALPHA, RUL_SCALE, delta_Value, FederatedHMM

ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH_PREFIX = ROOT_DIR / "c-mapss"

class customerDataset():
    def Import(self, FD):
        try:
            train_df = pd.read_csv(DATA_PATH_PREFIX / f"train_FD00{FD}.txt", sep=" ", header=None)
        except FileNotFoundError:
            print(f"Error: Could not find 'CMAPSS/train_FD00{FD}.txt'.")
            print("Please ensure the CMAPSS data is in a subdirectory named 'CMAPSS'.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
        train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3',
                            's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
                            's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                            's18', 's19', 's20', 's21']
        train_df = train_df.sort_values(['id', 'cycle'])

        test_df = pd.read_csv(DATA_PATH_PREFIX / f"test_FD00{FD}.txt", sep=" ", header=None)
        test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
        test_df.columns = train_df.columns
        test_df = test_df.sort_values(['id', 'cycle'])

        truth_df = pd.read_csv(DATA_PATH_PREFIX / f"RUL_FD00{FD}.txt", sep=" ", header=None)
        truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

        train_df['cycle_norm'] = train_df['cycle']
        cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL'])
        min_max_scaler = preprocessing.MinMaxScaler()
        norm_train_df = pd.DataFrame(
            min_max_scaler.fit_transform(train_df[cols_normalize]),
            columns=cols_normalize,
            index=train_df.index
        )
        join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
        train_df = join_df.reindex(columns=train_df.columns)

        test_df['cycle_norm'] = test_df['cycle']
        norm_test_df = pd.DataFrame(
            min_max_scaler.transform(test_df[cols_normalize]),
            columns=cols_normalize,
            index=test_df.index
        )
        test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
        test_df = test_join_df.reindex(columns=test_df.columns)
        test_df = test_df.reset_index(drop=True)

        rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        truth_df.columns = ['more']
        truth_df['id'] = truth_df.index + 1
        truth_df['max'] = truth_df['more']
        truth_df.drop('more', axis=1, inplace=True)

        selected_cols = ['id', 'cycle', 's2', 's3', 's4', 's7', 's8', 's9',
                         's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
        return train_df[selected_cols], test_df[selected_cols], truth_df

    def __gettrain__(self, FD='1'):
        train_df, _, _ = self.Import(FD)
        if train_df.empty:
            return np.array([])
        units = range(1, train_df['id'].nunique() + 1)
        multiUnitDataset = []
        for i in units:
            unit_data = train_df[train_df['id'] == i]
            signals = np.transpose(unit_data.iloc[:, 2:16].values)
            time = unit_data['cycle'].values
            T = max(time)
            multiUnitDataset.append(np.array([signals, time, T, i], dtype=object))
        return np.array(multiUnitDataset)

    def __gettest__(self, FD='1'):
        _, test_df, truth_df = self.Import(FD)
        if test_df.empty:
            return np.array([])
        units = range(1, test_df['id'].nunique() + 1)
        multiUnitDataset = []
        for i in units:
            unit_data = test_df[test_df['id'] == i]
            signals = np.transpose(unit_data.iloc[:, 2:16].values)
            time = unit_data['cycle'].values
            T = max(time) + truth_df[truth_df['id'] == i].iloc[0, 1]
            multiUnitDataset.append(np.array([signals, time, T, i], dtype=object))
        return np.array(multiUnitDataset)
    

def pinball_loss(output, target, quantiles=[ALPHA/2, 0.5, 1-ALPHA/2]):
    target = target.view(-1, 1)
    total_loss = 0.0
    for i, q in enumerate(quantiles):
        prediction = output[:, i].view(-1, 1)
        error = target - prediction
        loss = torch.mean(torch.max(q * error, (q - 1) * error))
        total_loss += loss
    return total_loss / len(quantiles)

class AllPreDegradationDataset(Dataset):
    def __init__(self, multi_unit_data, seq_len=50, stride=1):
        self.seq_len = seq_len
        self.stride = stride
        self.data = []
        if multi_unit_data is None:
            return
        for unit in multi_unit_data:
            signals = unit[0]
            time = unit[1]
            T = unit[2]
            signals = signals.transpose()
            n_steps = len(time)
            for i in range(0, n_steps - seq_len + 1, stride):
                label = T - time[i + seq_len - 1]
                label = label / RUL_SCALE
                self.data.append((signals[i:i+seq_len], label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        signal_sample, label = self.data[index]
        return torch.tensor(signal_sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class TestSlidingDataset(Dataset):
    def __init__(self, multi_unit_data, seq_len=50, stride=5):
        self.seq_len = seq_len
        self.stride = stride
        self.data = []
        self.unit_ids = []
        if multi_unit_data is None:
            return
        for unit_id, unit in enumerate(multi_unit_data):
            signals = unit[0]
            time = unit[1]
            T = unit[2]
            signals = signals.transpose()
            n_steps = len(time)
            for i in range(0, n_steps - seq_len + 1, stride):
                label = T - time[i + seq_len - 1]
                label = label / RUL_SCALE
                self.data.append((signals[i:i+seq_len], label))
                self.unit_ids.append(unit_id)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        signal_sample, label = self.data[index]
        unit_id = self.unit_ids[index]
        return torch.tensor(signal_sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), unit_id

def fedopt_train_quantile_model(model, client_train_data, client_df, seq_len,
                                batch_size=32, epochs_local=5, lr=0.002, rounds=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = model.to(device)
    global_model.train()
    
    client_df['id'] = client_df['id'].astype(int)
    client_weights = client_df['cluster'].value_counts().sort_index().to_numpy()
    total_weight = sum(client_weights)
    
    for r in range(rounds):
        local_state_dicts = []
        for i, client_data in enumerate(client_train_data):
            if not client_data:
                continue
            local_dataset = AllPreDegradationDataset(client_data, seq_len=seq_len, stride=1)
            if len(local_dataset) == 0:
                continue
            local_loader = DataLoader(local_dataset, batch_size=batch_size, shuffle=True)
            
            local_model = type(model)(input_size=14, num_layers=model.num_layers).to(device)
            local_model.load_state_dict(global_model.state_dict())
            local_model.train()
            
            optimizer = optim.Adam(local_model.parameters(), lr=lr)
            for epoch in range(epochs_local):
                for inputs, labels in local_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = local_model(inputs)
                    loss = pinball_loss(output, labels)
                    loss.backward()
                    optimizer.step()
            local_state_dicts.append(local_model.state_dict())
        
        if not local_state_dicts:
            print(f"Round {r+1} skipped: No data from any client.")
            continue

        global_state_dict = {}
        for key in global_model.state_dict().keys():
            weighted_sum = 0.0
            for i in range(len(local_state_dicts)):
                weight = client_weights[i]
                weighted_sum += weight * local_state_dicts[i][key].float()
            global_state_dict[key] = weighted_sum / total_weight
        global_model.load_state_dict(global_state_dict)
        print(f"FedOpt Round {r+1}/{rounds} completed.")
    
    return global_model

def split_data_among_clients(train_unit_data, client_df, num_clients=10):
    client_clusters = {row['id']: row['cluster'] for _, row in client_df.iterrows()}
    cluster_to_engines = {}
    for engine in train_unit_data:
        if isinstance(engine, np.ndarray) and len(engine) > 0:
            engine_id = int(engine[3])
        else:
            continue
        cluster_id = client_clusters.get(engine_id, -1)
        if cluster_id not in cluster_to_engines:
            cluster_to_engines[cluster_id] = []
        cluster_to_engines[cluster_id].append(engine)
    
    client_data = [[] for _ in range(num_clients)]
    for cluster_id in sorted(cluster_to_engines.keys()):
        client_sizes = [len(engines) for engines in client_data]
        target_client_idx = np.argmin(client_sizes)
        client_data[target_client_idx].extend(cluster_to_engines[cluster_id])
    return client_data


def compute_cqr_tdigest_with_scores(model, calibration_dataset, proxy_type='consistency',
                                     smoothing_W=3, batch_size=32):
    """
    proxy_type: 'consistency' -> |f(X_t) - (f(X_{t-1}) - 1/RUL_SCALE)|  
                'uncertainty' -> Q_hi(X_t) - Q_lo(X_t) 
    """
    loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
    digest = TDigest(delta=delta_Value)
    all_scores_full = []
    all_pred_medians = []
    all_pred_lowers = []
    all_pred_uppers = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            pred_lower = outputs[:, 0]
            pred_upper = outputs[:, 2]
            labels_flat = labels.view(-1)
            
            score_lower_diff = pred_lower - labels_flat
            score_upper_diff = labels_flat - pred_upper
            cqr_scores = torch.max(score_lower_diff, score_upper_diff)
            
            all_pred_medians.extend(outputs[:, 1].cpu().numpy().tolist())
            all_pred_lowers.extend(pred_lower.cpu().numpy().tolist())
            all_pred_uppers.extend(pred_upper.cpu().numpy().tolist())
            
            cqr_scores_np = cqr_scores.cpu().numpy()
            all_scores_full.extend(cqr_scores_np.tolist())
            for score in cqr_scores_np:
                digest.update(score)
    
    if proxy_type == 'consistency':
        raw_proxies = []
        for t in range(1, len(all_pred_medians)):
            proxy = abs(all_pred_medians[t] - (all_pred_medians[t - 1] - 1.0 / RUL_SCALE))
            raw_proxies.append(proxy)
        
        all_proxies = []
        for t in range(len(raw_proxies)):
            window = raw_proxies[max(0, t - smoothing_W + 1):t + 1]
            all_proxies.append(np.mean(window))
        
        all_scores_aligned = all_scores_full[1:]
    
    elif proxy_type == 'uncertainty':
        all_proxies = [all_pred_uppers[t] - all_pred_lowers[t] for t in range(len(all_pred_uppers))]
        all_scores_aligned = all_scores_full  
    
    else:
        raise ValueError("Proxy Error.")
    
    return digest, all_scores_full, all_scores_aligned, all_proxies


def compute_global_quantile_tdigest(global_digest, alpha=0.1):
    if global_digest.n == 0:
        return np.inf
    return global_digest.percentile(100 * (1 - alpha))

def _calculate_hmm_posteriors(client_scores, pi, A, emission_likelihood_func):
    n_obs = len(client_scores)
    num_states = A.shape[0]
    fwd, scaling_factors = np.zeros((n_obs, num_states)), np.zeros(n_obs)
    if n_obs == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0
    
    emission_probs_t0 = np.array([emission_likelihood_func(client_scores[0], m) for m in range(num_states)])
    fwd[0, :] = pi * emission_probs_t0
    scaling_factors[0] = np.sum(fwd[0, :])
    if scaling_factors[0] > 0:
        fwd[0, :] /= scaling_factors[0]

    for t in range(1, n_obs):
        emission_probs_t = np.array([emission_likelihood_func(client_scores[t], m) for m in range(num_states)])
        fwd[t, :] = np.dot(fwd[t - 1, :], A) * emission_probs_t
        scaling_factors[t] = np.sum(fwd[t, :])
        if scaling_factors[t] > 0:
            fwd[t, :] /= scaling_factors[t]

    bwd = np.zeros((n_obs, num_states))
    bwd[n_obs - 1, :] = 1.0
    for t in range(n_obs - 2, -1, -1):
        emission_probs_t1 = np.array([emission_likelihood_func(client_scores[t + 1], m) for m in range(num_states)])
        if scaling_factors[t+1] > 0:
            bwd[t, :] = np.dot(A, (emission_probs_t1 * bwd[t + 1, :])) / scaling_factors[t+1]
        else:
            bwd[t, :] = 1.0 / num_states

    eta = fwd * bwd
    row_sums = eta.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    eta /= row_sums
    log_likelihood = np.sum(np.log(scaling_factors + 1e-9))
    return eta, fwd, bwd, scaling_factors, log_likelihood


def run_local_e_step(client_proxies, global_pi, local_A, emission_likelihood_func, client_scores=None):
    n_obs = len(client_proxies)
    num_states = local_A.shape[0]
    eta, fwd, bwd, scaling_factors, log_likelihood = _calculate_hmm_posteriors(
        client_proxies, global_pi, local_A, emission_likelihood_func
    )
    if n_obs == 0:
        return {'pi_update': np.zeros(num_states),
                'A_transitions': np.zeros((num_states, num_states))}, \
               [TDigest(delta=delta_Value) for _ in range(num_states)], 0

    xi = np.zeros((n_obs - 1, num_states, num_states))
    for t in range(n_obs - 1):
        emission_probs_t1 = np.array([emission_likelihood_func(client_proxies[t + 1], m) for m in range(num_states)])
        denominator = np.dot(np.dot(fwd[t, :].T, local_A) * emission_probs_t1, bwd[t + 1, :])
        if denominator > 0:
            for i in range(num_states):
                numerator = fwd[t, i] * local_A[i, :] * emission_probs_t1 * bwd[t + 1, :]
                xi[t, i, :] = numerator / denominator

    pi_update = eta[0, :]
    A_transitions = np.sum(xi, axis=0)
    
    digest_values = client_scores if client_scores is not None else client_proxies
    client_state_digests = [TDigest(delta=delta_Value) for _ in range(num_states)]
    for t in range(n_obs):
        for m in range(num_states):
            if eta[t, m] > 1e-9:
                client_state_digests[m].update(digest_values[t], eta[t, m])
    
    return {'pi_update': pi_update, 'A_transitions': A_transitions}, client_state_digests, log_likelihood

def run_local_m_step(local_A_transitions, zeta):
    num_states = local_A_transitions.shape[0]
    A_new = np.zeros((num_states, num_states))
    for m in range(num_states - 1):
        N_mm = local_A_transitions[m, m]
        N_m_m1 = local_A_transitions[m, m+1]
        zeta_mm = zeta[m, 0]
        zeta_m_m1 = zeta[m, 1]
        denominator = (N_mm + N_m_m1) + (zeta_mm + zeta_m_m1) - 2
        if denominator > 1e-9:
            A_new[m, m] = (N_mm + zeta_mm - 1) / denominator
            A_new[m, m+1] = (N_m_m1 + zeta_m_m1 - 1) / denominator
        else:
            A_new[m, m] = A_new[m, m+1] = 0.5
    A_new[num_states - 1, num_states - 1] = 1.0
    return A_new

def predict_state_weights(score_context, global_pi, local_A, emission_likelihood_func):
    n_context = len(score_context)
    num_states = local_A.shape[0]
    if n_context == 0:
        return global_pi
    fwd = np.zeros((n_context, num_states))
    emission_probs_t0 = np.array([emission_likelihood_func(score_context[0], m) for m in range(num_states)])
    fwd[0, :] = global_pi * emission_probs_t0
    s = np.sum(fwd[0, :])
    if s > 0:
        fwd[0, :] /= s
    for t in range(1, n_context):
        emission_probs_t = np.array([emission_likelihood_func(score_context[t], m) for m in range(num_states)])
        fwd[t, :] = np.dot(fwd[t - 1, :], local_A) * emission_probs_t
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

def run_federated_hmm_training(NUM_HMM_STATES, NUM_EM_ROUNDS, CONVERGENCE_THRESHOLD,
                               client_proxy_sequences, all_cal_proxies,
                               client_score_sequences=None,
                               a_diag=0.8, zeta_val=10.0, random_seed=0):
    hmm = FederatedHMM(num_states=NUM_HMM_STATES, initial_scores=all_cal_proxies,
                       a_diag=a_diag, zeta_val=zeta_val, random_state=random_seed)
    num_clients = len(client_proxy_sequences)
    local_A_matrices = [hmm.A_init.copy() for _ in range(num_clients)]
    all_client_per_state_digests = [[] for _ in range(num_clients)]
    prev_log_likelihood = -np.inf
    num_states = hmm.num_states

    print(f"Starting Federated GEM for {NUM_HMM_STATES} states, {num_clients} clients.")

    for em_round in range(NUM_EM_ROUNDS):
        client_statistics = []
        total_log_likelihood = 0.0

        for i, client_proxy_seqs in enumerate(client_proxy_sequences):
            if not client_proxy_seqs:
                continue
            agg_pi_update = np.zeros(num_states)
            agg_A_transitions = np.zeros((num_states, num_states))
            agg_digests = [TDigest(delta=delta_Value) for _ in range(num_states)]
            client_ll = 0.0
            score_seqs = client_score_sequences[i] if client_score_sequences is not None else [None] * len(client_proxy_seqs)

            for seq_idx, seq_proxies in enumerate(client_proxy_seqs):
                if len(seq_proxies) <= 1:
                    continue
                seq_scores = score_seqs[seq_idx] if seq_idx < len(score_seqs) else None
                st, digests, ll = run_local_e_step(
                    seq_proxies, hmm.pi, local_A_matrices[i],
                    hmm.get_emission_likelihood, client_scores=seq_scores
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
            print("HMM training failed: no client statistics.")
            break
        
        proxy_only_digests = [[] for _ in range(num_clients)]
        if client_score_sequences is not None:
            for i, client_proxy_seqs in enumerate(client_proxy_sequences):
                if not client_proxy_seqs:
                    continue
                agg_proxy_digests = [TDigest(delta=delta_Value) for _ in range(num_states)]
                for seq_proxies in client_proxy_seqs:
                    if len(seq_proxies) <= 1:
                        continue
                    _, p_digests, _ = run_local_e_step(
                        seq_proxies, hmm.pi, local_A_matrices[i],
                        hmm.get_emission_likelihood, client_scores=None
                    )
                    for m in range(num_states):
                        agg_proxy_digests[m] += p_digests[m]
                proxy_only_digests[i] = agg_proxy_digests
        else:
            proxy_only_digests = all_client_per_state_digests

        hmm.global_m_step(client_statistics, proxy_only_digests)

        for st in client_statistics:
            ci = st['client_index']
            local_A_matrices[ci] = run_local_m_step(st['A_transitions'], hmm.zeta)

        print(f"  EM Round {em_round + 1}/{NUM_EM_ROUNDS} | Log-Likelihood = {total_log_likelihood:.8f}")
        if em_round > 0 and abs(total_log_likelihood - prev_log_likelihood) < CONVERGENCE_THRESHOLD:
            print(f"HMM training converged after {em_round + 1} rounds.")
            break
        prev_log_likelihood = total_log_likelihood
        
    if client_score_sequences is not None:
        for i, client_proxy_seqs in enumerate(client_proxy_sequences):
            if not client_proxy_seqs:
                continue
            score_seqs = client_score_sequences[i]
            agg_digests = [TDigest(delta=delta_Value) for _ in range(num_states)]
            for seq_idx, seq_proxies in enumerate(client_proxy_seqs):
                if len(seq_proxies) <= 1:
                    continue
                seq_scores = score_seqs[seq_idx] if seq_idx < len(score_seqs) else None
                _, digests, _ = run_local_e_step(
                    seq_proxies, hmm.pi, local_A_matrices[i],
                    hmm.get_emission_likelihood, client_scores=seq_scores
                )
                for m in range(num_states):
                    agg_digests[m] += digests[m]
            all_client_per_state_digests[i] = agg_digests

    return hmm, local_A_matrices, all_client_per_state_digests

def calculate_global_average_A(local_A_matrices):
    if not local_A_matrices:
        raise ValueError("Matrix A Error.")
    return np.sum(local_A_matrices, axis=0) / len(local_A_matrices)

def get_adaptive_quantile(rho_m, omega_k, all_client_digests, alpha):
    final_digest = TDigest(delta=delta_Value)
    num_clients = len(all_client_digests)
    num_states = len(rho_m)
    for k in range(num_clients):
        if not all_client_digests[k]:
            continue
        for m in range(num_states):
            weight = rho_m[m] * omega_k[k]
            if weight > 1e-9:
                scaled_digest = TDigest(delta=delta_Value)
                try:
                    for centroid in all_client_digests[k][m].centroids_to_list():
                        scaled_digest.update(centroid['m'], centroid['c'] * weight)
                    final_digest += scaled_digest
                except (ValueError, IndexError):
                    continue
    if final_digest.n == 0:
        return np.inf
    return final_digest.percentile(100 * (1 - alpha))

