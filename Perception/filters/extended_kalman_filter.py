# we suppose that input of radar are [depth,azimuth,velocity] =[r , teta, r°]
# and our state  estimation is x = [X,Y,Vx,Vy]

import matplotlib.pyplot as plt
import numpy as np
import math

class ExtendedKalmanFilter:
    def __init__(self, x0, P0, Q, R,dt):
        self.x = x0.astype(float)
        self.P = P0.astype(float)
        self.Q = Q.astype(float)
        self.R = R.astype(float)
        self.I = np.eye(x0.shape[0],dtype=np.float32)
        self.dt=dt

    def f(self, x, dt):
        X, Y, Vx, Vy = x

        return np.array([
            X + Vx * dt,
            Y + Vy * dt,
            Vx,
            Vy
        ], dtype=float)

    def jacobian_f(self, x, dt):
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

    def predict(self, dt):
        Jf = self.jacobian_f(self.x, dt)
        self.x = self.f(self.x, dt)
        self.P = Jf @ self.P @ Jf.T + self.Q
        return self.x


    def h(self, x):
        X, Y, Vx, Vy = x
        r = math.sqrt(X**2 + Y**2)

        if r < 1e-6:
            r = 1e-6

        return np.array([
            r,
            math.atan2(Y, X),
            (X * Vx + Y * Vy) / r
        ], dtype=float)
    



    def jacobian_h(self, x):
        X, Y, Vx, Vy = x
        r = math.sqrt(X**2 + Y**2)

        if r < 1e-6:
            r = 1e-6

        N = X * Vx + Y * Vy

        return np.array([
            [X / r, Y / r, 0, 0],
            [-Y / r**2, X / r**2, 0, 0],
            [
                Vx / r - (X * N) / r**3,
                Vy / r - (Y * N) / r**3,
                X / r,
                Y / r
            ]
        ], dtype=float)
    




    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def update(self, zk):
        zk = zk.astype(float)

        z_pred = self.h(self.x)
        y = zk - z_pred

        # bearing residual must be angle-normalized
        y[1] = self.normalize_angle(y[1])

        Jh = self.jacobian_h(self.x)

        S = Jh @ self.P @ Jh.T + self.R

        # K = P Jh.T S^{-1}
        K = np.linalg.solve(S.T, (self.P @ Jh.T).T).T

        I_KJh = self.I - K @ Jh

        self.x = self.x + K @ y

        # Joseph form
        self.P = I_KJh @ self.P @ I_KJh.T + K @ self.R @ K.T

        return self.x



def radar_to_state(z):  # [r , teta, r°] -> [X,Y,Vx,Vy]
    r, theta, r_dot = z

    X = r * np.cos(theta)
    Y = r * np.sin(theta)

    Vx = r_dot * np.cos(theta)
    Vy = r_dot * np.sin(theta)

    return np.array([X, Y, Vx, Vy], dtype=float)





class RadEKF:
    def __init__(self, measurement, R, dt=1/30):
        x0 = radar_to_state(measurement)

        P0 = np.diag([1, 1, 100, 100])
        Q = np.diag([0.1, 0.1, 1.0, 1.0])
        self.dt = dt
        self.ekf = ExtendedKalmanFilter(x0, P0, Q, R, self.dt)


    def predict(self):
        return self.ekf.predict(self.dt)
        

    def update(self, measurement):
        z = np.array(measurement, dtype=float)
        return self.ekf.update(z)
    

    def step(self, measurement):
        self.predict()
        return self.update(measurement)
    


