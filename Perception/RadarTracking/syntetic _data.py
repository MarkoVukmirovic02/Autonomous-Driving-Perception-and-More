
# Syntetic test we wanna see how good our filter is in controled enviroment.
# we will chose some initial state [X Y Vx Vy] transfer it into [r, teta , r*] add some noise
import numpy as np
import math
from EKF_radar import RadEKF
import matplotlib.pyplot as plt

x_true = np.array([30, 50, 9, 17], dtype=float)

def h_true(x):
    X, Y, Vx, Vy = x
    r = math.sqrt(X**2 + Y**2)
    return np.array([
        r,
        math.atan2(Y, X),
        (X * Vx + Y * Vy) / r
    ], dtype=float)

R = np.diag([
    2.5**2,     # range noise std = 0.5 m
    0.15**2,    # angle noise std = 0.05 rad
    0.7**2      # radial velocity noise std = 0.3 m/s
])
phi = 0.0

def f_true(x, dt):
    global phi

    X, Y, Vx, Vy = x

    omega = 0.5
    speed = np.sqrt(Vx**2 + Vy**2)

    phi += omega * dt

    Vx = speed * np.cos(phi)
    Vy = speed * np.sin(phi)

    X += Vx * dt
    Y += Vy * dt

    return np.array([X, Y, Vx, Vy])

N=100
dt=1/30

true_states = []
noisy_measurements = []
ekf_estimates = []

x_true = np.array([30, 50, 9, 17], dtype=float)

R_true = np.diag([2.5**2, 0.15**2, 0.7**2])
R_ekf = R_true.copy()

z0 = h_true(x_true) + np.random.multivariate_normal(np.zeros(3), R_true)
tracker = RadEKF(z0, R_ekf, dt)

for k in range(N):
    # 1. propagate true hidden state
    x_true = f_true(x_true, dt)

    # 2. generate ideal radar measurement from true state
    z_true = h_true(x_true)

    # 3. add sensor noise
    noise = np.random.multivariate_normal(np.zeros(3), R_true)
    z_noisy = z_true + noise

    # 4. EKF uses only noisy radar
    x_est = tracker.step(z_noisy)

    # 5. save for plotting
    true_states.append(x_true.copy())
    noisy_measurements.append(z_noisy.copy())
    ekf_estimates.append(x_est.copy())


radar_xy = np.array([
    [z[0] * np.cos(z[1]), z[0] * np.sin(z[1])]
    for z in noisy_measurements
])

true_states = np.array(true_states)
ekf_estimates = np.array(ekf_estimates)
noisy_measurements = np.array(noisy_measurements)



plt.plot(true_states[:, 0], true_states[:, 1], label="True trajectory")
plt.scatter(radar_xy[:, 0], radar_xy[:, 1], s=10, label="Noisy radar")
plt.plot(ekf_estimates[:, 0], ekf_estimates[:, 1], label="EKF estimate")
plt.legend()
plt.axis("equal")
plt.grid(True)
import os

plt.savefig(
    os.path.expanduser("~/Desktop/Autonomna_Voznja/solution_yugo45/PHOTO OBRADA/EKF/plots/ekf_result_circle.png"),
    dpi=300,
    bbox_inches="tight"
)



