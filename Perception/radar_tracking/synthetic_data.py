import math
import os

import matplotlib.pyplot as plt
import numpy as np

from EKF_radar import RadEKF
from UKF_radar import UnscentedKalmanFilter


# ============================================================
# Experiment configuration
# ============================================================

SEED = 42
N = 100
dt = 1.0 / 30.0

rng = np.random.default_rng(SEED)

initial_true_state = np.array(
    [30.0, 50.0, 9.0, 17.0],
    dtype=float,
)


# Actual measurement covariance used by the simulated radar.
R_true = np.diag([
    2.5**2,     # range standard deviation = 2.5 m
    0.15**2,    # bearing standard deviation = 0.15 rad
    0.7**2,     # range-rate standard deviation = 0.7 m/s
])

# Both filters are told the correct radar covariance.
R_ekf = R_true.copy()
R_ukf = R_true.copy()


# ============================================================
# Angle utility
# ============================================================

def normalize_angle(angle):
    """
    Map an angle into the interval [-pi, pi].
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ============================================================
# Radar measurement model
#
# State:
#     x = [X, Y, Vx, Vy]
#
# Measurement:
#     z = [range, bearing, range_rate]
# ============================================================

def radar_measurement_model(x):
    X, Y, Vx, Vy = x

    range_squared = X**2 + Y**2
    r = math.sqrt(max(range_squared, 1e-12))

    bearing = math.atan2(Y, X)
    range_rate = (X * Vx + Y * Vy) / r

    return np.array(
        [r, bearing, range_rate],
        dtype=float,
    )


# Same function under the old name, if desired.
h_true = radar_measurement_model


# ============================================================
# Constant-velocity model used internally by the UKF
#
# Important:
# The true object turns, while the filters assume constant
# velocity. Therefore, this experiment tests model mismatch.
# ============================================================

def constant_velocity_model(x, dt):
    X, Y, Vx, Vy = x

    return np.array([
        X + Vx * dt,
        Y + Vy * dt,
        Vx,
        Vy,
    ], dtype=float)


# ============================================================
# True turning motion
# ============================================================

omega = 0.5

# Preserve the direction of the initial velocity.
phi = math.atan2(
    initial_true_state[3],
    initial_true_state[2],
)


def propagate_true_state(x, dt):
    """
    Propagate the target using a constant-speed turning model.

    The target turns with angular velocity omega, but both filters
    currently assume constant Cartesian velocity.
    """
    global phi

    X, Y, Vx, Vy = x

    speed = math.sqrt(Vx**2 + Vy**2)

    phi += omega * dt

    Vx_new = speed * math.cos(phi)
    Vy_new = speed * math.sin(phi)

    X_new = X + Vx_new * dt
    Y_new = Y + Vy_new * dt

    return np.array([
        X_new,
        Y_new,
        Vx_new,
        Vy_new,
    ], dtype=float)


# ============================================================
# Convert the first radar measurement into an initial state
# ============================================================

def radar_measurement_to_state(z):
    """
    Construct an approximate Cartesian state from one radar
    measurement.

    A single radar measurement does not reveal tangential velocity.
    Therefore, this initialization assumes that the initial velocity
    is entirely radial.
    """
    r, bearing, range_rate = z

    X = r * math.cos(bearing)
    Y = r * math.sin(bearing)

    Vx = range_rate * math.cos(bearing)
    Vy = range_rate * math.sin(bearing)

    return np.array([
        X,
        Y,
        Vx,
        Vy,
    ], dtype=float)


# ============================================================
# Generate one shared synthetic dataset
# ============================================================

true_states = []
noisy_measurements = []

x_true = initial_true_state.copy()

for k in range(N):
    # 1. Propagate the hidden true state.
    x_true = propagate_true_state(x_true, dt)

    # 2. Produce the ideal radar measurement.
    z_true = radar_measurement_model(x_true)

    # 3. Add radar noise.
    noise = rng.multivariate_normal(
        mean=np.zeros(3),
        cov=R_true,
    )

    z_noisy = z_true + noise

    # Keep bearing inside a valid interval.
    z_noisy[1] = normalize_angle(z_noisy[1])

    true_states.append(x_true.copy())
    noisy_measurements.append(z_noisy.copy())


true_states = np.asarray(true_states)
noisy_measurements = np.asarray(noisy_measurements)


# ============================================================
# Initialize EKF
# ============================================================

z0 = noisy_measurements[0].copy()

ekf_tracker = RadEKF(
    z0,
    R_ekf,
    dt,
)


# ============================================================
# Initialize UKF
# ============================================================

x0_ukf = radar_measurement_to_state(z0)

P0_ukf = np.diag([
    10.0**2,    # X uncertainty
    10.0**2,    # Y uncertainty
    8.0**2,     # Vx uncertainty
    8.0**2,     # Vy uncertainty
])


# Process noise for a constant-velocity model driven by
# unknown Cartesian acceleration.
acceleration_std = 4.0
acceleration_variance = acceleration_std**2

G = np.array([
    [0.5 * dt**2, 0.0],
    [0.0, 0.5 * dt**2],
    [dt, 0.0],
    [0.0, dt],
])

Q_ukf = G @ (
    acceleration_variance * np.eye(2)
) @ G.T


# Adjust this constructor only if your UKF class uses another
# parameter ordering or keyword names.
ukf_tracker = UnscentedKalmanFilter(
    x0=x0_ukf,
    P0=P0_ukf,
    Q=Q_ukf,
    R=R_ukf,
    f=constant_velocity_model,
    h=radar_measurement_model,
    alpha=0.3,
    beta=2.0,
    kappa=0.0,
)


# ============================================================
# Run both filters on exactly the same measurements
# ============================================================

ekf_estimates = []
ukf_estimates = []

for z_noisy in noisy_measurements[1:]:
    x_ekf = ekf_tracker.step(z_noisy)

    x_ukf, P_ukf = ukf_tracker.step(
        z_noisy,
        dt,
    )

    ekf_estimates.append(
        np.asarray(x_ekf).reshape(-1).copy()
    )

    ukf_estimates.append(
        x_ukf.copy()
    )

ekf_estimates = np.asarray(ekf_estimates)
ukf_estimates = np.asarray(ukf_estimates)

evaluation_true_states = true_states[1:]


# ============================================================
# Convert radar polar measurements to Cartesian points
# ============================================================

radar_xy = np.column_stack([
    noisy_measurements[:, 0] *
    np.cos(noisy_measurements[:, 1]),

    noisy_measurements[:, 0] *
    np.sin(noisy_measurements[:, 1]),
])


# ============================================================
# Evaluation
# ============================================================

def rmse(reference, estimate):
    error = reference - estimate
    return np.sqrt(np.mean(error**2, axis=0))


def position_rmse(reference, estimate):
    squared_distance = np.sum(
        (reference[:, :2] - estimate[:, :2])**2,
        axis=1,
    )

    return np.sqrt(np.mean(squared_distance))



ekf_state_rmse = rmse(
    evaluation_true_states,
    ekf_estimates,
)

ukf_state_rmse = rmse(
    evaluation_true_states,
    ukf_estimates,
)

ekf_position_rmse = position_rmse(
    evaluation_true_states,
    ekf_estimates,
)

ukf_position_rmse = position_rmse(
    evaluation_true_states,
    ukf_estimates,
)


print("\nRMSE per state component")
print("------------------------")
print(
    "EKF [X, Y, Vx, Vy]:",
    np.round(ekf_state_rmse, 4),
)
print(
    "UKF [X, Y, Vx, Vy]:",
    np.round(ukf_state_rmse, 4),
)

print("\nTwo-dimensional position RMSE")
print("--------------------------------")
print(f"EKF: {ekf_position_rmse:.4f} m")
print(f"UKF: {ukf_position_rmse:.4f} m")


# ============================================================
# Trajectory plot
# ============================================================

plt.figure(figsize=(10, 8))

plt.plot(
    true_states[:, 0],
    true_states[:, 1],
    label="True trajectory",
    linewidth=2,
)

plt.scatter(
    radar_xy[:, 0],
    radar_xy[:, 1],
    s=12,
    alpha=0.5,
    label="Noisy radar",
)

plt.plot(
    ekf_estimates[:, 0],
    ekf_estimates[:, 1],
    label=(
        f"EKF estimate "
        f"(RMSE={ekf_position_rmse:.2f} m)"
    ),
)

plt.plot(
    ukf_estimates[:, 0],
    ukf_estimates[:, 1],
    label=(
        f"UKF estimate "
        f"(RMSE={ukf_position_rmse:.2f} m)"
    ),
)

plt.xlabel("X position [m]")
plt.ylabel("Y position [m]")
plt.title("Radar tracking: EKF versus UKF")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()

output_path = os.path.expanduser(
    "~/Desktop/Autonomna_Voznja/"
    "solution_yugo45/PHOTO OBRADA/"
    "EKF/plots/ekf_vs_ukf_circle.png"
)

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True,
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"\nPlot saved to:\n{output_path}")