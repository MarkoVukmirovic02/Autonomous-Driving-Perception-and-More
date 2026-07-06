from CameraPerception.yolov8n import YOLOPerception
from CameraTracking.kalman_filter import KalmanFilter
import numpy as np



def xyxy_to_cxcywh(bbox):
    x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
    cx = (x1 + x2)/2
    cy = (y1 + y2)/2
    w  = x2 - x1
    h  = y2 - y1
    return [cx,cy,w,h]

class BoxKalmanFilter:
    def __init__(self, bbox):


        measurement = xyxy_to_cxcywh(bbox)

        P0 = np.diag([10, 10, 10, 10, 100, 100, 100, 100]).astype(np.float32)

        R = (np.eye(4) * 5).astype(np.float32)

        dt=1/30

        F = np.array([
            [1,0,0,0,dt,0,0,0],
            [0,1,0,0,0,dt,0,0],
            [0,0,1,0,0,0,dt,0],
            [0,0,0,1,0,0,0,dt],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
            ], dtype=np.float32)

        Q = (np.eye(8) * 0.01).astype(np.float32)

        x0 = np.array([
            measurement[0],
            measurement[1],
            measurement[2],
            measurement[3],
            0, 0, 0, 0
        ], dtype=np.float32).reshape(8, 1)

        self.kf = KalmanFilter(x0, P0, F, Q, R)


    def predict(self,dt):
        self.kf.set_dt(dt)
        return self.kf.predict()
        


    def update(self,bbox):
        z = np.array(xyxy_to_cxcywh(bbox), dtype=np.float32).reshape(4, 1)
        return self.kf.update(z)

    def predict_k_steps(self, k):
        future_x = self.kf.x.copy()
        F = self.kf.F.copy()

        for _ in range(k):
            future_x = F @ future_x

        return future_x
