from .types import GaussianState, validate_gaussian_state
import numpy as np


def common_to_ctrv_low_speed(x, P, destination_state):
    return _low_speed_heading_conversion(
        x,
        P,
        destination_state,
        include_acceleration=False,
    )


def common_to_ctra_low_speed(x, P, destination_state):
    return _low_speed_heading_conversion(
        x,
        P,
        destination_state,
        include_acceleration=True,
    )

def _low_speed_heading_conversion(
    common_mean,
    common_covariance,
    destination_state,
    *,
    include_acceleration,
):
    # Validate the common Cartesian distribution.
    common = validate_gaussian_state(
        common_mean,
        common_covariance,
        6,
        (),
    )

    # Native orders:
    # CTRV: [px, py, v, yaw, yaw_rate]
    # CTRA: [px, py, v, a, yaw, yaw_rate]
    if include_acceleration:
        native_dimension = 6
        yaw_index = 4
        rate_index = 5
    else:
        native_dimension = 5
        yaw_index = 3
        rate_index = 4

    destination = validate_gaussian_state(
        destination_state.x,
        destination_state.P,
        native_dimension,
        (yaw_index,),
    )

    px, py, vx, vy, ax, ay = common.x
    yaw = destination.x[yaw_index]
    yaw_rate = destination.x[rate_index]

    c = np.cos(yaw)
    s = np.sin(yaw)

    signed_speed = vx * c + vy * s
    acceleration = ax * c + ay * s

    speed_heading_derivative = -vx * s + vy * c
    acceleration_heading_derivative = -ax * s + ay * c

    # Augmented coordinates:
    # [px, py, vx, vy, ax, ay, yaw, yaw_rate]
    augmented_covariance = np.zeros((8, 8), dtype=float)
    augmented_covariance[:6, :6] = common.P

    retained_indices = [yaw_index, rate_index]
    augmented_covariance[6:, 6:] = destination.P[
        np.ix_(retained_indices, retained_indices)
    ]

    J = np.zeros((native_dimension, 8), dtype=float)

    J[0, 0] = 1.0
    J[1, 1] = 1.0

    # Signed speed projected onto the retained heading.
    J[2, 2] = c
    J[2, 3] = s
    J[2, 6] = speed_heading_derivative

    if include_acceleration:
        mean = np.array([
            px, py, signed_speed, acceleration, yaw, yaw_rate
        ])

        J[3, 4] = c
        J[3, 5] = s
        J[3, 6] = acceleration_heading_derivative
    else:
        mean = np.array([
            px, py, signed_speed, yaw, yaw_rate
        ])

    J[yaw_index, 6] = 1.0
    J[rate_index, 7] = 1.0

    covariance = J @ augmented_covariance @ J.T
    covariance = 0.5 * (covariance + covariance.T)

    return validate_gaussian_state(
        mean,
        covariance,
        native_dimension,
        (yaw_index,),
    )






def cv_to_common_mean(x):
    px, py, vx, vy = np.asarray(x, dtype=float)
    return np.array([px, py, vx, vy, 0.0, 0.0])

def cv_to_common_covariance(P, acceleration_covariance):
    P = np.asarray(P, dtype=float)
    acceleration_covariance = np.asarray(
        acceleration_covariance, dtype=float
    )

    if P.shape != (4, 4):
        raise ValueError("CV covariance must have shape (4, 4).")

    if acceleration_covariance.shape != (2, 2):
        raise ValueError(
            "Acceleration covariance must have shape (2, 2)."
        )

    common_covariance = np.zeros((6, 6), dtype=float)

    common_covariance[:4, :4] = P
    common_covariance[4:, 4:] = acceleration_covariance

    return common_covariance


def ca_to_common_mean(x):
    px, py, vx, vy, ax, ay = np.asarray(x, dtype=float)
    return np.array([px, py, vx, vy, ax, ay])


def ca_to_common_covariance(P):
    P = np.asarray(P, dtype=float)

    if P.shape != (6, 6):
        raise ValueError("CA covariance must have shape (6, 6).")

    return P.copy()


def ctrv_to_common_mean(x):
    px, py, v, yaw, yaw_rate = np.asarray(x, dtype=float)

    c = np.cos(yaw)
    s = np.sin(yaw)

    vx = v * c
    vy = v * s

    ax = -v * yaw_rate * s
    ay =  v * yaw_rate * c

    return np.array([px, py, vx, vy, ax, ay])

def ctrv_to_common_covariance(
    x, P, longitudinal_acceleration_variance
):
    px, py, v, yaw, yaw_rate = np.asarray(x, dtype=float)
    P = np.asarray(P, dtype=float)
    variance = float(longitudinal_acceleration_variance)

    if P.shape != (5, 5):
        raise ValueError("CTRV covariance must have shape (5, 5).")

    if not np.isfinite(variance) or variance < 0:
        raise ValueError(
            "Longitudinal acceleration variance must be "
            "finite and nonnegative."
        )

    c = np.cos(yaw)
    s = np.sin(yaw)

    J = np.array([
        [1, 0, 0,             0,                 0],
        [0, 1, 0,             0,                 0],
        [0, 0, c,            -v * s,             0],
        [0, 0, s,             v * c,             0],
        [0, 0, -yaw_rate * s, -v * yaw_rate * c, -v * s],
        [0, 0,  yaw_rate * c, -v * yaw_rate * s,  v * c],
    ], dtype=float)

    direction = np.array([0, 0, 0, 0, c, s], dtype=float)

    common_covariance = (
        J @ P @ J.T
        + variance * np.outer(direction, direction)
    )

    return 0.5 * (common_covariance + common_covariance.T)

def ctra_to_common_mean(x):
    px, py, v, acceleration, yaw, yaw_rate = np.asarray(
        x, dtype=float
    )

    c = np.cos(yaw)
    s = np.sin(yaw)

    vx = v * c
    vy = v * s

    ax = acceleration * c - v * yaw_rate * s
    ay = acceleration * s + v * yaw_rate * c

    return np.array([px, py, vx, vy, ax, ay])


def ctra_to_common_covariance(x, P):
    px, py, v, acceleration, yaw, yaw_rate = np.asarray(
        x, dtype=float
    )
    P = np.asarray(P, dtype=float)

    if P.shape != (6, 6):
        raise ValueError("CTRA covariance must have shape (6, 6).")

    c = np.cos(yaw)
    s = np.sin(yaw)

    J = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, c, 0, -v * s, 0],
        [0, 0, s, 0,  v * c, 0],
        [
            0, 0, -yaw_rate * s, c,
            -acceleration * s - v * yaw_rate * c,
            -v * s
        ],
        [
            0, 0, yaw_rate * c, s,
            acceleration * c - v * yaw_rate * s,
            v * c
        ],
    ], dtype=float)

    common_covariance = J @ P @ J.T

    return 0.5 * (common_covariance + common_covariance.T)


def common_to_cv_mean(x):
    x = np.asarray(x, dtype=float)

    if x.shape != (6,):
        raise ValueError("Common mean must have shape (6,).")

    return x[:4].copy()


def common_to_cv_covariance(P):
    P = np.asarray(P, dtype=float)

    if P.shape != (6, 6):
        raise ValueError("Common covariance must have shape (6, 6).")

    return P[:4, :4].copy()


def common_to_ca_mean(x):
    x = np.asarray(x, dtype=float)

    if x.shape != (6,):
        raise ValueError("Common mean must have shape (6,).")

    return x.copy()


def common_to_ca_covariance(P):
    P = np.asarray(P, dtype=float)

    if P.shape != (6, 6):
        raise ValueError("Common covariance must have shape (6, 6).")

    return P.copy()


def _common_motion_terms(x, minimum_speed):
    x = np.asarray(x, dtype=float)

    if x.shape != (6,) or not np.all(np.isfinite(x)):
        raise ValueError(
            "Common mean must be a finite vector of shape (6,)."
        )

    minimum_speed = float(minimum_speed)

    if not np.isfinite(minimum_speed) or minimum_speed <= 0:
        raise ValueError("minimum_speed must be finite and positive.")

    px, py, vx, vy, ax, ay = x

    speed = np.hypot(vx, vy)

    if speed < minimum_speed:
        raise ValueError(
            "Speed is too low to reconstruct heading and turn rate."
        )

    yaw = np.arctan2(vy, vx)
    acceleration = (vx * ax + vy * ay) / speed
    yaw_rate = (vx * ay - vy * ax) / speed**2

    return x, speed, yaw, acceleration, yaw_rate


def common_to_ctrv_mean(x, *, minimum_speed):
    x, speed, yaw, acceleration, yaw_rate = _common_motion_terms(
        x, minimum_speed
    )

    return np.array([
        x[0], x[1], speed, yaw, yaw_rate
    ])


def common_to_ctra_mean(x, *, minimum_speed):
    x, speed, yaw, acceleration, yaw_rate = _common_motion_terms(
        x, minimum_speed
    )

    return np.array([
        x[0], x[1], speed, acceleration, yaw, yaw_rate
    ])



def _common_to_ctra_jacobian(x, minimum_speed):
    x, speed, yaw, acceleration, yaw_rate = _common_motion_terms(
        x, minimum_speed
    )

    px, py, vx, vy, ax, ay = x
    speed_squared = speed**2

    J = np.zeros((6, 6), dtype=float)

    J[0, 0] = 1.0
    J[1, 1] = 1.0

    # Speed
    J[2, 2] = vx / speed
    J[2, 3] = vy / speed

    # Longitudinal acceleration
    J[3, 2] = (
        ax / speed - acceleration * vx / speed_squared
    )
    J[3, 3] = (
        ay / speed - acceleration * vy / speed_squared
    )
    J[3, 4] = vx / speed
    J[3, 5] = vy / speed

    # Heading
    J[4, 2] = -vy / speed_squared
    J[4, 3] =  vx / speed_squared

    # Turn rate
    J[5, 2] = (ay - 2 * yaw_rate * vx) / speed_squared
    J[5, 3] = (-ax - 2 * yaw_rate * vy) / speed_squared
    J[5, 4] = -vy / speed_squared
    J[5, 5] =  vx / speed_squared

    return J


def common_to_ctrv_covariance(x, P, *, minimum_speed):
    P = np.asarray(P, dtype=float)

    if P.shape != (6, 6):
        raise ValueError("Common covariance must have shape (6, 6).")

    J = _common_to_ctra_jacobian(x, minimum_speed)
    J = J[[0, 1, 2, 4, 5], :]

    native_covariance = J @ P @ J.T

    return 0.5 * (native_covariance + native_covariance.T)


def common_to_ctra_covariance(x, P, *, minimum_speed):
    P = np.asarray(P, dtype=float)

    if P.shape != (6, 6):
        raise ValueError("Common covariance must have shape (6, 6).")

    J = _common_to_ctra_jacobian(x, minimum_speed)

    native_covariance = J @ P @ J.T

    return 0.5 * (native_covariance + native_covariance.T)
