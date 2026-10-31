import numpy as np

from Perception.experiments.imm_setup import build_imm
from Perception.tracking.imm import IMM


def assert_same_state(actual, expected, angle_indices):
    difference = actual.x - expected.x

    for index in angle_indices:
        difference[index] = (
            difference[index] + np.pi
        ) % (2 * np.pi) - np.pi

    np.testing.assert_allclose(
        difference,
        np.zeros_like(difference),
        rtol=0.0,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        actual.P,
        expected.P,
        rtol=1e-10,
        atol=1e-10,
    )


def check_single_model(index, name, measurements, dt):
    # Independent filter instances with identical initial conditions.
    adapter = build_imm().filters[index]
    reference_filter = build_imm().filters[index].filter

    imm = IMM(
        filters=[adapter],
        mode_probabilities=[1.0],
        transition_matrix=[[1.0]],
    )

    # First verify that self-only mixing changes nothing.
    before = adapter.state_snapshot()

    predicted_probabilities = imm.predict_mode_probabilities()
    weights = imm.mixing_probabilities(predicted_probabilities)
    mixed = imm.mix_states(weights)[0]

    np.testing.assert_array_equal(mixed.x, before.x)
    np.testing.assert_array_equal(mixed.P, before.P)

    # Mixing calculations must not mutate the filter.
    after = adapter.state_snapshot()
    np.testing.assert_array_equal(after.x, before.x)
    np.testing.assert_array_equal(after.P, before.P)

    for z in measurements:
        # Compare prediction.
        imm.predict(dt)
        reference_filter.predict(dt)

        assert_same_state(
            adapter.state_snapshot(),
            reference_filter.state_snapshot(),
            adapter.state_angle_indices,
        )

        # Compare measurement update.
        imm.update(z)
        reference_filter.update(z)

        assert_same_state(
            adapter.state_snapshot(),
            reference_filter.state_snapshot(),
            adapter.state_angle_indices,
        )

        np.testing.assert_allclose(
            imm.mode_probabilities,
            [1.0],
            rtol=0.0,
            atol=1e-14,
        )

    print(f"PASS: {name} — self-mixing, prediction and update")


def main():
    dt = 0.2
    rng = np.random.default_rng(42)

    # Shared noisy position observations for straight motion at 8 m/s.
    times = dt * np.arange(1, 31)
    true_positions = np.column_stack([
        8.0 * times,
        np.zeros_like(times),
    ])

    measurements = true_positions + rng.normal(
        0.0, 0.7, size=true_positions.shape
    )

    names = [
        "KF-CV",
        "KF-CA",
        "EKF-CTRV",
        "UKF-CTRA",
    ]

    for index, name in enumerate(names):
        check_single_model(index, name, measurements, dt)

    print("All single-model equivalence checks passed.")


if __name__ == "__main__":
    main()