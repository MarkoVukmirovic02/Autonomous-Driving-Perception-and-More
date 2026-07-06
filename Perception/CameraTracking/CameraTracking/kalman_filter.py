import numpy as np


class KalmanFilter:


    def __init__(self, x0, P0, F, Q, R):



        self.x = x0      
        self.P = P0
        self.F = F
        self.Q = Q


        self.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
                 ],dtype=np.float32)
        

        self.R = R
        self.I = np.eye(x0.shape[0],dtype=np.float32)


    def set_dt(self,dt):
        self.F = np.array([
        [1,0,0,0,dt,0,0,0],
        [0,1,0,0,0,dt,0,0],
        [0,0,1,0,0,0,dt,0],
        [0,0,0,1,0,0,0,dt],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1]
        ], dtype=np.float32)


    def predict(self):

        self.x=self.F @ self.x

        self.P=self.F @ self.P@ self.F.T + self.Q

        return self.x
    
    def update(self,zk):

        y=zk- self.H @ self.x

        S= self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        I_KH = (self.I - K @ self.H)

        self.x= self.x + K @ y

        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x
    