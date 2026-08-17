#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common import *


def main():
    # default value for N_REPS is 10, to reduce the running time, adjust N_REPS = 1 for short version
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

    df = pd.DataFrame(summary_rows)
    raw_path = os.path.join(outputs_dir, "simulation_scenario_b_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw per-rep results saved to {raw_path}")

    agg_df = _aggregate_results(df)
    agg_path = os.path.join(outputs_dir, "simulation_scenario_b_summary.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Aggregated summary saved to {agg_path}")

    return df


if __name__ == "__main__":
    results_df = main()
