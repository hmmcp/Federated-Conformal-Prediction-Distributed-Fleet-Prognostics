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
    print("EXPERIMENT C: Violation of shared-emission assumption")
    print("=" * 70)

    for delta in [0.1, 0.3, 0.5, 1.0]:
        dgp_C = copy.deepcopy(base_dgp)
        dgp_C['fleet_emission_shift'] = delta
        for use_fed in [True, False]:
            label = "Federated" if use_fed else "Centralized"
            print(f"  emission_shift={delta}, {label} ...")

            for rep in range(N_REPS):
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

    df = pd.DataFrame(summary_rows)
    raw_path = os.path.join(outputs_dir, "simulation_scenario_c_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw per-rep results saved to {raw_path}")

    agg_df = _aggregate_results(df)
    agg_path = os.path.join(outputs_dir, "simulation_scenario_c_summary.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Aggregated summary saved to {agg_path}")

    return df


if __name__ == "__main__":
    results_df = main()
