from CameraPerception.yolov8n import YOLOPerception
from CameraTracking.box_tracker import BoxKalmanFilter

from risk.risk_engine import (
    compute_action_risk,
    box_iou,
    compute_ego_danger,
)


class PerceptionRiskModule:
    def __init__(
        self,
        model_path="yolov8n.pt",
        conf=0.10,
        imgsz=640,
        predict_ahead=1,
    ):
        self.perception = YOLOPerception(
            model_path=model_path,
            conf=conf,
            imgsz=imgsz,
            allowed_classes={
                "person", "bicycle", "car", "motorcycle", "bus", "truck"
            }
        )

        self.trackable_classes = {
            "car", "truck", "bus", "motorcycle", "bicycle", "person"
        }

        self.kalman_filters = {}
        self.pending_predictions = {}
        self.iou_scores = []

        self.frame_idx = 0

        self.predict_ahead = predict_ahead



    def update_tracks_and_predict(self, detections):
        target_frame = self.frame_idx + self.predict_ahead
        dt=1/30


        for det in detections:
            if det["class"] not in self.trackable_classes:
                continue

            track_id = det.get("track_id")
            if track_id is None:
                continue


            if track_id not in self.kalman_filters:
                self.kalman_filters[track_id] = BoxKalmanFilter(det["bbox"])
                continue

            kf = self.kalman_filters[track_id]
            kf.predict(dt=dt)
            kf.update(det["bbox"])
                
            future_state = kf.predict_k_steps(self.predict_ahead)

            future_cx = future_state[0, 0]
            future_cy = future_state[1, 0]
            future_w = future_state[2, 0]
            future_h = future_state[3, 0]

            det["pred_box"] = [
                    int(future_cx - future_w / 2),
                    int(future_cy - future_h / 2),
                    int(future_cx + future_w / 2),
                    int(future_cy + future_h / 2),
                    ]

            if target_frame not in self.pending_predictions:
                self.pending_predictions[target_frame] = []

            self.pending_predictions[target_frame].append({
                    "track_id": track_id,
                    "pred_box": det["pred_box"],
                    "source_frame": self.frame_idx,
                    })

        return detections
            

    def evaluate_old_predictions(self, detections):
        if self.frame_idx not in self.pending_predictions:
            return

        for prediction in self.pending_predictions[self.frame_idx]:
            pred_track_id = prediction["track_id"]
            pred_box = prediction["pred_box"]

            for det in detections:
                if det.get("track_id") == pred_track_id:
                    true_box = det["bbox"]
                    iou = box_iou(pred_box, true_box)
                    self.iou_scores.append(iou)
                    break

        del self.pending_predictions[self.frame_idx]

    def process_frame(self, frame):
        detections, results = self.perception.detect_and_track(frame)

        self.evaluate_old_predictions(detections)

        detections = self.update_tracks_and_predict(detections)

        frame_h, frame_w = frame.shape[:2]

        best_action, action_risks = compute_action_risk(
            detections=detections,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        danger_data = compute_ego_danger(
            detections=detections,
            frame_w=frame_w,
            frame_h=frame_h,
        )

        self.frame_idx += 1

        return {
            "best_action": best_action,
            "action_risks": action_risks,
            "detections": detections,
            "results": results,
            "avg_prediction_iou": (
                sum(self.iou_scores) / len(self.iou_scores)
                if len(self.iou_scores) > 0 else None
            ),

            "ego_box": danger_data["ego_box"],
            "max_ego_iou": danger_data["max_ego_iou"],
            "danger": danger_data["danger"],
            "danger_threshold": danger_data["danger_threshold"],
            "danger_objects": danger_data["danger_objects"],
        }