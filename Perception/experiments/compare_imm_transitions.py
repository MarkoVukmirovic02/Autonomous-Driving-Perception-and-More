from pathlib import Path
import csv

import numpy as np

from Perception.experiments.imm_stop_restart import run


def main():
    output = (
        Path(__file__).resolve().parent / "imm_transition_results"
    )
    output.mkdir(parents=True, exist_ok=True)

    seeds = [2026, 2027, 2028]
    rows = []

    for persistence in [0.90, 0.95, 0.98, 0.995]:
        matrix = np.full((4, 4), (1.0 - persistence) / 3.0)
        np.fill_diagonal(matrix, persistence)

        position_squared_errors = []
        speed_squared_errors = []
        dropout_errors = []
        switches = []

        for seed in seeds:
            folder = output / f"stay_{persistence}" / f"seed_{seed}"

            run(
                seed=seed,
                output=folder,
                transition_matrix=matrix,
            )

            with np.load(folder / "stop_restart.npz") as data:
                estimates = data["estimates"]
                truth = data["truth"]
                probabilities = data["probabilities"]
                dropout = data["dropout"].astype(bool)

                position_error = np.linalg.norm(
                    estimates[:, :2] - truth[:, :2],
                    axis=1,
                )

                estimated_speed = np.linalg.norm(
                    estimates[:, 2:4], axis=1
                )
                speed_error = estimated_speed - truth[:, 2]

                position_squared_errors.extend(position_error**2)
                speed_squared_errors.extend(speed_error**2)
                dropout_errors.extend(position_error[dropout])

                dominant = np.argmax(probabilities, axis=1)
                switches.append(np.count_nonzero(np.diff(dominant)))

        row = {
            "persistence": persistence,
            "position_rmse_m": float(
                np.sqrt(np.mean(position_squared_errors))
            ),
            "speed_rmse_m_s": float(
                np.sqrt(np.mean(speed_squared_errors))
            ),
            "max_dropout_position_error_m": float(
                np.max(dropout_errors)
            ),
            "mean_dominant_mode_switches": float(np.mean(switches)),
        }
        rows.append(row)

    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nTRANSITION-MATRIX COMPARISON")
    for row in rows:
        print(
            f"stay={row['persistence']:.3f} | "
            f"position RMSE={row['position_rmse_m']:.3f} m | "
            f"speed RMSE={row['speed_rmse_m_s']:.3f} m/s | "
            f"max dropout error="
            f"{row['max_dropout_position_error_m']:.3f} m | "
            f"dominant-mode switches="
            f"{row['mean_dominant_mode_switches']:.1f}"
        )

    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()