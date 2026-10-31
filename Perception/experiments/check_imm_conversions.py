import numpy as np

from Perception.tracking.state_conversion import (
    ctrv_to_common_mean,
    ctrv_to_common_covariance,
    ctra_to_common_mean,
    ctra_to_common_covariance,
    common_to_ctrv_mean,
    common_to_ctrv_covariance,
    common_to_ctra_mean,
    common_to_ctra_covariance,
)


MINIMUM_SPEED = 0.5


def numerical_jacobian(function, x, output_angles=()):
    x = np.asarray(x, dtype=float)
    output = function(x)
    J = np.empty((output.size, x.size))

    for column in range(x.size):
        step = 1e-5 * max(1.0, abs(x[column]))

        plus = x.copy()
        minus = x.copy()
        plus[column] += step
        minus[column] -= step

        difference = function(plus) - function(minus)

        for index in output_angles:
            difference[index] = (
                difference[index] + np.pi
            ) % (2 * np.pi) - np.pi

        J[:, column] = difference / (2 * step)

    return J


def assert_valid_covariance(P):
    assert np.all(np.isfinite(P)), "Nonfinite covariance"

    np.testing.assert_allclose(
        P, P.T, rtol=0.0, atol=1e-10
    )

    eigenvalues = np.linalg.eigvalsh(P)
    tolerance = 1e-10 * max(1.0, np.max(np.abs(eigenvalues)))

    assert eigenvalues.min() >= -tolerance, (
        f"Covariance is not PSD: minimum eigenvalue "
        f"{eigenvalues.min()}"
    )


def check_covariance_map(
    name, mean_map, covariance_map, x, rng, output_angles=()
):
    J = numerical_jacobian(mean_map, x, output_angles)
    dimension = x.size

    # Rank-one inputs isolate individual derivative columns.
    covariances = []
    for column in range(dimension):
        direction = np.eye(dimension)[:, column]
        covariances.append(np.outer(direction, direction))

    # Correlated inputs also exercise cross terms.
    for _ in range(5):
        A = rng.normal(size=(dimension, dimension))
        covariances.append(A @ A.T / dimension)

    for P in covariances:
        actual = covariance_map(x, P)
        expected = J @ P @ J.T

        np.testing.assert_allclose(
            actual, expected,
            rtol=2e-5,
            atol=2e-7,
            err_msg=f"{name}: covariance transformation mismatch",
        )

        assert_valid_covariance(actual)


def assert_same_mean(actual, expected, angle_indices=()):
    difference = actual - expected

    for index in angle_indices:
        difference[index] = (
            difference[index] + np.pi
        ) % (2 * np.pi) - np.pi

    np.testing.assert_allclose(
        difference, np.zeros_like(difference),
        rtol=0.0, atol=1e-9,
    )


def main():
    rng = np.random.default_rng(123)

    def to_ctrv(y):
        return common_to_ctrv_mean(
            y, minimum_speed=MINIMUM_SPEED
        )

    def to_ctrv_covariance(y, P):
        return common_to_ctrv_covariance(
            y, P, minimum_speed=MINIMUM_SPEED
        )

    def to_ctra(y):
        return common_to_ctra_mean(
            y, minimum_speed=MINIMUM_SPEED
        )

    def to_ctra_covariance(y, P):
        return common_to_ctra_covariance(
            y, P, minimum_speed=MINIMUM_SPEED
        )

    # Positive speeds permit native mean round-trip comparisons.
    # Include zero turn rate and headings near the angular boundary.
    for yaw in [0.0, 0.8, -1.7, np.pi - 1e-4]:
        for yaw_rate in [0.0, 0.3, -0.24]:
            ctrv = np.array([12., -4., 8., yaw, yaw_rate])
            ctra = np.array([12., -4., 8., 1.3, yaw, yaw_rate])

            common_ctrv = ctrv_to_common_mean(ctrv)
            common_ctra = ctra_to_common_mean(ctra)

            check_covariance_map(
                "CTRV -> common",
                ctrv_to_common_mean,
                lambda x, P: ctrv_to_common_covariance(
                    x, P, 0.0
                ),
                ctrv,
                rng,
            )

            check_covariance_map(
                "CTRA -> common",
                ctra_to_common_mean,
                ctra_to_common_covariance,
                ctra,
                rng,
            )

            check_covariance_map(
                "Common -> CTRV",
                to_ctrv,
                to_ctrv_covariance,
                common_ctra,
                rng,
                output_angles=(3,),
            )

            check_covariance_map(
                "Common -> CTRA",
                to_ctra,
                to_ctra_covariance,
                common_ctra,
                rng,
                output_angles=(4,),
            )

            assert_same_mean(
                to_ctrv(common_ctrv), ctrv, angle_indices=(3,)
            )
            assert_same_mean(
                to_ctra(common_ctra), ctra, angle_indices=(4,)
            )

    print("PASS: four nonlinear covariance maps")
    print("PASS: positive-speed native mean round trips")

    # Verify the additional CTRV longitudinal acceleration prior.
    ctrv = np.array([12., -4., 8., 0.8, 0.3])
    P = np.diag([1., 1., 0.4, 0.02, 0.01])
    variance = 0.8**2

    without_prior = ctrv_to_common_covariance(ctrv, P, 0.0)
    with_prior = ctrv_to_common_covariance(ctrv, P, variance)

    direction = np.array([
        0., 0., 0., 0.,
        np.cos(ctrv[3]),
        np.sin(ctrv[3]),
    ])

    np.testing.assert_allclose(
        with_prior - without_prior,
        variance * np.outer(direction, direction),
        rtol=1e-10,
        atol=1e-10,
    )
    assert_valid_covariance(with_prior)

    print("PASS: CTRV acceleration-prior contribution")

    # Negative native speed is mapped to a nonnegative-speed convention.
    # Check physical common-state equivalence, not native parameter equality.
    for native, forward, reverse in [
        (
            np.array([12., -4., -8., 0.8, 0.3]),
            ctrv_to_common_mean,
            to_ctrv,
        ),
        (
            np.array([12., -4., -8., 1.3, 0.8, 0.3]),
            ctra_to_common_mean,
            to_ctra,
        ),
    ]:
        common = forward(native)
        recovered = reverse(common)

        assert recovered[2] >= 0
        assert_same_mean(forward(recovered), common)

    print("PASS: negative-speed physical mean equivalence")

    # Current policy: reject reverse conversion near zero speed.
    stopped = np.zeros(6)

    for reverse in [to_ctrv, to_ctra]:
        try:
            reverse(stopped)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Low-speed conversion should raise ValueError."
            )

    print("PASS: low-speed rejection")
    print("All conversion checks passed.")


if __name__ == "__main__":
    main()