#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Model: Quantile Regression
* Method 0: No CP
* Method 1: Proposed HMM based method
* Method 2: Lu et al

Proxies:
  - consistency: |f(X_t) - (f(X_{t-1}) - 1/RUL_SCALE)|  (smoothed)
  - uncertainty: Q_hi(X_t) - Q_lo(X_t)  (predicted interval width)
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tdigest import TDigest
from torch.utils.data import DataLoader
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Data folder
DATA_DIR = ROOT_DIR / "c-mapss"

# Output folders
MODEL_DIR = ROOT_DIR / "Models"
OUTPUT_DIR = ROOT_DIR / "Outputs"

# Create folders automatically if needed
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


from models import ALPHA, RUL_SCALE, delta_Value, QuantileLSTM
from functions import (
    customerDataset,
    AllPreDegradationDataset,
    TestSlidingDataset,
    fedopt_train_quantile_model,
    split_data_among_clients,
    compute_cqr_tdigest_with_scores,
    run_federated_hmm_training,
    calculate_global_average_A,
    predict_state_weights,
    get_likelihood_ratio_weights,
    get_adaptive_quantile,
)

plt.rcParams.update({'font.size': 20})

SEQ_LEN = 50
BATCH_SIZE = 32
NUM_EM_ROUNDS = 20
CONVERGENCE_THRESHOLD = 1e-9
num_clients = 5
Tcw = 20

SUMMARY_COLUMNS = [
    "FD", "Baseline", "CP Method", "Proxy", "Num Clients",
    "Num States", "Zeta", "A_diag", "Num Layers",
    "Random Seed", "Cal Ratio",
    "Target Coverage",
    "Coverage_Overall", "Width_Overall",
    "Coverage_lt50", "Width_lt50",
    "Coverage_50_100", "Width_50_100",
    "Coverage_gt100", "Width_gt100",
]

#if not os.path.exists(summary_file):
#    pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(summary_file, index=False)

# Hyperparameter grid
NUM_HMM_STATES = 5
ZETA_VAL = 5
A_DIAG = 0.8
RANDOM_SEED = 10000
CAL_RATIO = 0.2
NUM_LAYERS = 2
PROXY_TYPE = 'uncertainty'

for FD in ['1']:
    customer = customerDataset()
    train_unit_data = customer.__gettrain__(FD)
    test_unit_data = customer.__gettest__(FD)
    num_ids = train_unit_data.shape[0]

    model_tag = f"FD{FD}_NL{NUM_LAYERS}_CSR{CAL_RATIO}_C{num_clients}_RS{RANDOM_SEED}"
    model_save_path = MODEL_DIR / f"QR_{model_tag}.pth"

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    guaranteed_assignments = np.arange(num_clients)
    remaining_assignments = np.random.randint(0, num_clients, size=num_ids - num_clients)
    combined_assignments = np.concatenate([guaranteed_assignments, remaining_assignments])
    np.random.shuffle(combined_assignments)

    client_df = pd.DataFrame({
        'id': np.arange(1, num_ids + 1),
        'cluster': combined_assignments
    })
    client_train_data_full = split_data_among_clients(train_unit_data, client_df, num_clients=num_clients)

    client_train_proper_data = [[] for _ in range(num_clients)]
    client_cal_data = [[] for _ in range(num_clients)]
    for k in range(num_clients):
        client_engines = client_train_data_full[k]
        if not client_engines:
            continue
        try:
            engines_train, engines_cal = train_test_split(
                client_engines, test_size=CAL_RATIO, random_state=0
            )
            client_train_proper_data[k] = engines_train
            client_cal_data[k] = engines_cal
        except ValueError:
            client_train_proper_data[k] = client_engines

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

    model = QuantileLSTM(input_size=14, num_layers=NUM_LAYERS)
    if not os.path.exists(model_save_path):
        print(f"No saved model found. Starting federated model training...{model_tag}")
        trained_model = fedopt_train_quantile_model(
            model, client_train_proper_data, client_df,
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
            epochs_local=5, lr=0.002, rounds=20
        )
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"Model saved: {model_save_path}")
    else:
        print(f"Loading already trained federated model...{model_tag}")
        trained_model = QuantileLSTM(input_size=14, num_layers=NUM_LAYERS)
        trained_model.load_state_dict(
            torch.load(model_save_path, map_location=torch.device("cpu"))
        )
    
    trained_model = trained_model.to(device)
    trained_model.eval()
    test_dataset = TestSlidingDataset(test_unit_data, seq_len=SEQ_LEN, stride=1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_data_by_engine = {}
    trained_model.eval()
    with torch.no_grad():
        for inputs, labels, unit_ids in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = trained_model(inputs)
            pred_lower_raw = outputs[:, 0]
            pred_median = outputs[:, 1]
            pred_upper_raw = outputs[:, 2]
            labels_flat = labels.view(-1)
            score_lower_diff = pred_lower_raw - labels_flat
            score_upper_diff = labels_flat - pred_upper_raw
            cqr_scores = torch.max(score_lower_diff, score_upper_diff)

            for i, uid in enumerate(unit_ids):
                uid = uid.item()
                if uid not in test_data_by_engine:
                    test_data_by_engine[uid] = []
                test_data_by_engine[uid].append({
                    'score': cqr_scores[i].item(),
                    'pred_lower_raw': pred_lower_raw[i].item(),
                    'pred_median': pred_median[i].item(),
                    'pred_upper_raw': pred_upper_raw[i].item(),
                    'label': labels[i].item()
                })

    
    client_score_sequences = [[] for _ in range(num_clients)]
    client_proxy_sequences = [[] for _ in range(num_clients)]
    all_cal_scores = []
    all_cal_scores_full = []
    all_cal_proxies = []

    for k in range(num_clients):
        client_engines = client_cal_data[k]
        if not client_engines:
            continue
        for engine in client_engines:
            engine_dataset = AllPreDegradationDataset([engine], seq_len=SEQ_LEN, stride=1)
            if len(engine_dataset) == 0:
                continue
            _, scores_full, scores_aligned, proxies = compute_cqr_tdigest_with_scores(
                trained_model, engine_dataset, proxy_type=PROXY_TYPE
            )
            if scores_aligned and proxies:
                client_score_sequences[k].append(np.array(scores_aligned))
                client_proxy_sequences[k].append(np.array(proxies))
                all_cal_scores.extend(scores_aligned)
                all_cal_proxies.extend(proxies)
            all_cal_scores_full.extend(scores_full)

    N_cal = len(all_cal_scores_full)
    target_rank = np.ceil((1 - ALPHA) * (N_cal + num_clients))
    p_level = target_rank / N_cal if N_cal > 0 else 2.0
    if not all_cal_scores_full or p_level > 1.0:
        q_static = np.inf
    else:
        q_static = np.quantile(all_cal_scores_full, p_level, method='higher')

    for engine_id in test_data_by_engine:
        engine_data = test_data_by_engine[engine_id]
        if PROXY_TYPE == 'consistency':
            engine_data[0]['proxy'] = 0.0
            for t in range(1, len(engine_data)):
                engine_data[t]['proxy'] = abs(
                    engine_data[t]['pred_median'] - (engine_data[t-1]['pred_median'] - 1.0/RUL_SCALE)
                )
                
            W = 3
            raw = [d['proxy'] for d in engine_data]
            for t in range(len(engine_data)):
                window = raw[max(0, t - W + 1):t + 1]
                engine_data[t]['proxy'] = np.mean(window)
        elif PROXY_TYPE == 'uncertainty':
            for t in range(len(engine_data)):
                engine_data[t]['proxy'] = (
                    engine_data[t]['pred_upper_raw'] - engine_data[t]['pred_lower_raw']
                )

    config_tag = (f"{model_tag}_P{PROXY_TYPE}_M{NUM_HMM_STATES}"
                    f"_Z{ZETA_VAL}_A{A_DIAG}")
    print(f"\n Configuration: {config_tag}")

    hmm, final_A_matrices, all_client_per_state_digests = run_federated_hmm_training(
        NUM_HMM_STATES, NUM_EM_ROUNDS, CONVERGENCE_THRESHOLD,
        client_proxy_sequences, all_cal_proxies,
        client_score_sequences=client_score_sequences,
        a_diag=A_DIAG, zeta_val=ZETA_VAL, random_seed=RANDOM_SEED
    )
    client_overall_digests = [TDigest(delta=delta_Value) for _ in range(num_clients)]
    global_cal_digest_hmm = TDigest(delta=delta_Value)
    for i, seq_list in enumerate(client_proxy_sequences):
        for seq in seq_list:
            for s in seq:
                client_overall_digests[i].update(s)
                global_cal_digest_hmm.update(s)
    client_pdfs = [hmm._build_pdf_from_digest(d) for d in client_overall_digests]
    global_pdf = hmm._build_pdf_from_digest(global_cal_digest_hmm)

    true_values, pred_medians_list = [], []
    intervals_static, intervals_adaptive, intervals_raw = [], [], []
    avg_A_matrix = calculate_global_average_A(final_A_matrices)

    for engine_id in sorted(test_data_by_engine.keys()):
        engine_data = test_data_by_engine[engine_id]
        if len(engine_data) < 2:
            continue
        context_window = [d['proxy'] for d in engine_data[max(0, len(engine_data)-Tcw):-1]]
        last_entry = engine_data[-1]
        pl = last_entry['pred_lower_raw']
        pm = last_entry['pred_median']
        pu = last_entry['pred_upper_raw']
        true_rul = last_entry['label']

        intervals_raw.append((pl, pu))
        intervals_static.append((pl - q_static, pu + q_static))

        rho_m = predict_state_weights(context_window, hmm.pi, avg_A_matrix, hmm.get_emission_likelihood)
        omega_k = get_likelihood_ratio_weights(context_window, client_pdfs, global_pdf)
        q_adaptive = get_adaptive_quantile(rho_m, omega_k, all_client_per_state_digests, alpha=ALPHA)
        intervals_adaptive.append((pl - q_adaptive, pu + q_adaptive))

        true_values.append(true_rul)
        pred_medians_list.append(pm)

    true_values = np.array(true_values) * RUL_SCALE
    pred_medians_arr = np.array(pred_medians_list) * RUL_SCALE
    intervals_raw = np.array(intervals_raw) * RUL_SCALE
    intervals_static = np.array(intervals_static) * RUL_SCALE
    intervals_adaptive = np.array(intervals_adaptive) * RUL_SCALE

    def compute_metrics(true_vals, intervals):
        lower = intervals[:, 0].copy()
        upper = intervals[:, 1].copy()
        lower[lower < 0] = 0
        cov = np.mean((true_vals >= lower) & (true_vals <= upper))
        wid = np.mean(upper - lower)
        wid_sd = np.std(upper - lower)
        return cov, wid, wid_sd

    def compute_grouped_metrics(true_vals, intervals):
        lower = intervals[:, 0].copy()
        upper = intervals[:, 1].copy()
        lower[lower < 0] = 0
        covered = (true_vals >= lower) & (true_vals <= upper)
        widths = upper - lower
        
        results = {}
        bins_def = [(-np.inf, 50, 'lt50'), (50, 100, '50_100'), (100, np.inf, 'gt100')]
        for lo, hi, label in bins_def:
            mask = (true_vals > lo) & (true_vals <= hi)
            if label == 'lt50':
                mask = true_vals <= 50
            elif label == 'gt100':
                mask = true_vals > 100
            if np.sum(mask) > 0:
                results[f'Coverage_{label}'] = np.mean(covered[mask])
                results[f'Width_{label}'] = np.mean(widths[mask])
            else:
                results[f'Coverage_{label}'] = np.nan
                results[f'Width_{label}'] = np.nan
        return results

    for method_name, method_intervals in [
        ('No CP', intervals_raw),
        ('Lu et al', intervals_static),
        ('HMM', intervals_adaptive)
    ]:
        cov_all, wid_all, _ = compute_metrics(true_values, method_intervals)
        grouped = compute_grouped_metrics(true_values, method_intervals)

        row = {
            "FD": FD, "Baseline": "QR", "CP Method": method_name,
            "Proxy": PROXY_TYPE, "Num Clients": num_clients,
            "Num States": NUM_HMM_STATES, "Zeta": ZETA_VAL,
            "A_diag": A_DIAG, "Num Layers": NUM_LAYERS,
            "Random Seed": RANDOM_SEED, "Cal Ratio": CAL_RATIO,
            "Target Coverage": 100 * (1 - ALPHA),
            "Coverage_Overall": cov_all * 100,
            "Width_Overall": wid_all,
        }
        row.update(grouped)
        for k_name in ['Coverage_lt50', 'Coverage_50_100', 'Coverage_gt100']:
            if not np.isnan(row.get(k_name, np.nan)):
                row[k_name] = row[k_name] * 100

    print(f"  Raw:      Cov={compute_metrics(true_values, intervals_raw)[0]*100:.1f}%, "
            f"W={compute_metrics(true_values, intervals_raw)[1]:.1f}, "
            f"std={compute_metrics(true_values, intervals_raw)[2]:.1f}")
    print(f"  Lu et al: Cov={compute_metrics(true_values, intervals_static)[0]*100:.1f}%, "
            f"W={compute_metrics(true_values, intervals_static)[1]:.1f}, "
            f"std={compute_metrics(true_values, intervals_static)[2]:.1f}")
    print(f"  HMM:      Cov={compute_metrics(true_values, intervals_adaptive)[0]*100:.1f}%, "
            f"W={compute_metrics(true_values, intervals_adaptive)[1]:.1f}, "
            f"std={compute_metrics(true_values, intervals_adaptive)[2]:.1f}")

    sorted_indices = np.argsort(true_values)
    tv_s = true_values[sorted_indices]
    pm_s = pred_medians_arr[sorted_indices]

    df_results = pd.DataFrame({
        'true_rul': true_values,

        'lower_raw': intervals_raw[:, 0],
        'upper_raw': intervals_raw[:, 1],

        'lower_static': intervals_static[:, 0],
        'upper_static': intervals_static[:, 1],

        'lower_adaptive': intervals_adaptive[:, 0],
        'upper_adaptive': intervals_adaptive[:, 1],
    })

    df_results.loc[df_results['lower_raw'] < 0, 'lower_raw'] = 0
    df_results.loc[df_results['lower_static'] < 0, 'lower_static'] = 0
    df_results.loc[df_results['lower_adaptive'] < 0, 'lower_adaptive'] = 0

    df_results['covered_raw'] = (
        (df_results['true_rul'] >= df_results['lower_raw']) &
        (df_results['true_rul'] <= df_results['upper_raw'])
    )
    df_results['width_raw'] = df_results['upper_raw'] - df_results['lower_raw']

    df_results['covered_static'] = (
        (df_results['true_rul'] >= df_results['lower_static']) &
        (df_results['true_rul'] <= df_results['upper_static'])
    )
    df_results['width_static'] = df_results['upper_static'] - df_results['lower_static']

    df_results['covered_adaptive'] = (
        (df_results['true_rul'] >= df_results['lower_adaptive']) &
        (df_results['true_rul'] <= df_results['upper_adaptive'])
    )
    df_results['width_adaptive'] = df_results['upper_adaptive'] - df_results['lower_adaptive']

    bins = [-np.inf, 50, 100, np.inf]
    group_labels = ['< 50', '50-100', '> 100']

    df_results['rul_group'] = pd.cut(
        df_results['true_rul'],
        bins=bins,
        labels=group_labels,
        right=True
    )

    grouped_stats = df_results.groupby('rul_group', observed=True)[
        ['covered_raw', 'width_raw',
            'covered_static', 'width_static',
            'covered_adaptive', 'width_adaptive']
    ].mean()

    # make sure all groups appear even if one group is empty
    grouped_stats = grouped_stats.reindex(group_labels)

    n_groups = len(grouped_stats)
    x_indices = np.arange(n_groups)
    bar_width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # --- Plot 1: Empirical Coverage ---
    ax1.bar(
        x_indices - bar_width, grouped_stats['covered_raw'],
        width=bar_width, label='Raw QR', color='green', alpha=0.7
    )
    ax1.bar(
        x_indices, grouped_stats['covered_static'],
        width=bar_width, label='Lu et al. FCP', color='blue', alpha=0.7
    )
    ax1.bar(
        x_indices + bar_width, grouped_stats['covered_adaptive'],
        width=bar_width, label='Proposed Method', color='red', alpha=0.7
    )

    ax1.axhline(
        y=1 - ALPHA, color='black', linestyle='--',
        label=f'Target Coverage ({(1-ALPHA)*100:.0f}%)'
    )

    ax1.set_xlabel('True RUL Group')
    ax1.set_ylabel('Empirical Coverage')
    ax1.set_title('Empirical Coverage')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(grouped_stats.index)
    ax1.legend(loc='lower left')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.05)

    # --- Plot 2: Average Interval Width ---
    ax2.bar(
        x_indices - bar_width, grouped_stats['width_raw'],
        width=bar_width, label='Raw QR', color='green', alpha=0.7
    )
    ax2.bar(
        x_indices, grouped_stats['width_static'],
        width=bar_width, label='Lu et al. FCP', color='blue', alpha=0.7
    )
    ax2.bar(
        x_indices + bar_width, grouped_stats['width_adaptive'],
        width=bar_width, label='Proposed Method', color='red', alpha=0.7
    )

    ax2.set_xlabel('True RUL Group')
    ax2.set_ylabel('Average Interval Width')
    ax2.set_title('Average Interval Width')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(grouped_stats.index)
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"Combined_Metrics_QR_{config_tag}.pdf",
        dpi=300
    )
    plt.show()

    plt.figure(figsize=(14, 7))
    x_axis = np.arange(len(tv_s))
    plt.plot(x_axis, tv_s, 'k--', label="True RUL")
    plt.plot(x_axis, pm_s, 'b-', label="Predicted Median RUL")

    for label, arr, color, alpha_val in [
        ("Lu et al FCP", intervals_static, 'blue', 0.2),
        ("Proposed", intervals_adaptive, 'red', 0.3),
        ("Raw QR", intervals_raw, 'green', 0.2),
    ]:
        lo_s = arr[sorted_indices, 0].copy()
        hi_s = arr[sorted_indices, 1]
        lo_s[lo_s < 0] = 0
        c, w, *_ = compute_metrics(true_values, arr)
        plt.fill_between(x_axis, lo_s, hi_s, color=color, alpha=alpha_val,
                            label=f"{label} (Cov={c*100:.1f}%, W={w:.1f})")

    plt.xlabel("Engine Index (sorted by True RUL)")
    plt.ylabel("RUL (Normalized)")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.savefig(
        OUTPUT_DIR / f"Plot_QR_{config_tag}.pdf",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

print("========== All processing complete ==========")
