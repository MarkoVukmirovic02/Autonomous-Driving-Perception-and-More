import numpy as np


class UnscentedKalmanFilter:
    def __init__(self, x0, P0, Q, R, f, h,measurement_dim=3, alpha=1e-3, beta=2.0, kappa=0.0):
        self.x = x0.astype(float)
        self.P = P0.astype(float)
        self.Q = Q.astype(float)
        self.R = R.astype(float)
        self.m=measurement_dim

        self.f = f
        self.h = h

        self.n = self.x.shape[0]

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n

        self.wm, self.wc = self._compute_weights()

    def _compute_weights(self):
        wm = np.zeros(2 * self.n + 1)
        wc = np.zeros(2 * self.n + 1)

        wm[0] = self.lambda_ / (self.n + self.lambda_)
        wc[0] = wm[0] + (1 - self.alpha**2 + self.beta)

        wm[1:] = 1 / (2 * (self.n + self.lambda_))
        wc[1:] = 1 / (2 * (self.n + self.lambda_))

        return wm, wc

    def generate_sigma_points(self):
        sigma_points = np.zeros((2 * self.n + 1, self.n))

        sigma_points[0] = self.x

        A = np.linalg.cholesky((self.n + self.lambda_) * self.P)

        for i in range(self.n):
            sigma_points[i + 1] = self.x + A[:, i]
            sigma_points[i + 1 + self.n] = self.x - A[:, i]

        return sigma_points
        
        
    def predict_sigma_points(self, sigma_points, dt):

        predicted_sigma = np.zeros_like(sigma_points)

        for i in range(2 * self.n + 1):
            predicted_sigma[i] = self.f(sigma_points[i], dt)

        return predicted_sigma
    

    def predict(self,dt):
        sigma_points = self.generate_sigma_points()

        predicted_sigma=self.predict_sigma_points(sigma_points,dt)
        
        x_pred=np.zeros(self.n)

        for i in range(2*self.n+1):
            x_pred+=self.wm[i] * predicted_sigma[i]

        P_pred=np.zeros((self.n,self.n))

        for i in range(2*self.n+1):
            dx = predicted_sigma[i] - x_pred
            P_pred += self.wc[i] * np.outer(dx, dx)
        
        P_pred+=self.Q

        self.x=x_pred
        self.P=P_pred
        self.predicted_sigma = predicted_sigma

        return self.x, self.P
        
    def measurement_sigma_points(self,predicted_sigma):

        measurement_sigma = np.zeros((2 * self.n + 1, self.m))
        for i in range(2*self.n+1):
            measurement_sigma[i]= self.h(predicted_sigma[i])

        return measurement_sigma

    def predicted_measurement(self):
        measurement_sigma = self.measurement_sigma_points(self.predicted_sigma)

        z_pred = np.zeros(self.m)

        for i in range(2 * self.n + 1):
            z_pred += self.wm[i] * measurement_sigma[i]

        S = np.zeros((self.m, self.m))

        for i in range(2 * self.n + 1):
            dz = measurement_sigma[i] - z_pred
            S += self.wc[i] * np.outer(dz, dz)

        S += self.R

        Pxz = np.zeros((self.n, self.m))

        for i in range(2 * self.n + 1):
            dx = self.predicted_sigma[i] - self.x
            dz = measurement_sigma[i] - z_pred
            Pxz += self.wc[i] * np.outer(dx, dz)

        return z_pred, S, Pxz
    

    def update(self,z):

        z=z.astype(float)

        z_pred,S,Pxz = self.predicted_measurement()

        y=z-z_pred

        K=np.linalg.solve(S.T,Pxz.T).T

        self.x=self.x + K@y

        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return self.x,self.P
    

    def step(self, z,dt):
        self.predict(dt)
        return self.update(z)
    

