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

    df = pd.DataFrame(summary_rows)
    raw_path = os.path.join(outputs_dir, "simulation_scenario_a_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw per-rep results saved to {raw_path}")

    agg_df = _aggregate_results(df)
    agg_path = os.path.join(outputs_dir, "simulation_scenario_a_summary.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Aggregated summary saved to {agg_path}")

    return df


if __name__ == "__main__":
    results_df = main()
