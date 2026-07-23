import numpy as np

from Perception.camera_tracking.yolov8n import YOLOPerception
from Perception.camera_tracking.box_tracker import BoxKalmanFilter
from Perception.camera_tracking.risk_engine import (
    compute_action_risk,
    box_iou,
    compute_ego_danger,
)


class PerceptionRiskModule:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.10,
        imgsz: int = 640,
        predict_ahead: int = 1,
    ) -> None:

        
        if not isinstance(predict_ahead, int):
            raise TypeError("predict_ahead must be an integer.")


        if predict_ahead < 1:
            raise ValueError("predict_ahead must be at least 1.")

        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf must be between 0 and 1.")

        if not isinstance(imgsz, int) or imgsz <= 0:
            raise ValueError("imgsz must be a positive integer.")

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

    @staticmethod
    def _state_to_bbox(
        state: np.ndarray,
    ) -> list[int]:
        """
        Convert a filter state beginning with

            [cx, cy, width, height, ...]

        into an xyxy bounding box:

            [x1, y1, x2, y2]
        """
        state = np.asarray(
            state,
            dtype=float,
        ).reshape(-1)

        if state.size < 4:
            raise ValueError(
                "state must contain at least "
                "[cx, cy, width, height]."
            )

        if not np.all(np.isfinite(state[:4])):
            raise ValueError(
                "Bounding-box state contains NaN or infinite values."
            )

        cx = float(state[0])
        cy = float(state[1])
        width = max(0.0, float(state[2]))
        height = max(0.0, float(state[3]))

        return [
            int(round(cx - width / 2.0)),
            int(round(cy - height / 2.0)),
            int(round(cx + width / 2.0)),
            int(round(cy + height / 2.0)),
        ]

    def update_tracks_and_predict(
        self,
        detections,
        dt: float = 1.0 / 30.0,
    ):

        dt = float(dt)

        if not np.isfinite(dt):
            raise ValueError("dt must be finite.")

        if dt < 0.0:
            raise ValueError("dt must be non-negative.")

        target_frame = self.frame_idx + self.predict_ahead

        for det in detections:
            if det["class"] not in self.trackable_classes:
                continue

            track_id = det.get("track_id")

            if track_id is None:
                continue

            if track_id not in self.kalman_filters:
                self.kalman_filters[track_id] = BoxKalmanFilter(
                    det["bbox"]
                )

                det["pred_box"] = list(det["bbox"])
                continue

            tracker = self.kalman_filters[track_id]

            tracker.step(
                bbox=det["bbox"],
                dt=dt,
            )

            future_state = tracker.predict_k_steps(
                k=self.predict_ahead,
                dt=dt,
            )

            # New generic KF state is a flat array, shape (8,).
            det["pred_box"] = self._state_to_bbox(future_state)

            self.pending_predictions.setdefault(
                target_frame,
                [],
            ).append(
                {
                    "track_id": track_id,
                    "pred_box": det["pred_box"],
                    "source_frame": self.frame_idx,
                }
            )

        return detections
            

    def evaluate_old_predictions(
        self,
        detections,
    ) -> None:
        predictions = self.pending_predictions.pop(
            self.frame_idx,
            None,
        )

        if predictions is None:
            return

        detections_by_track_id = {
            det.get("track_id"): det
            for det in detections
            if det.get("track_id") is not None
        }

        for prediction in predictions:
            detection = detections_by_track_id.get(
                prediction["track_id"]
            )

            if detection is None:
                continue

            iou = box_iou(
                prediction["pred_box"],
                detection["bbox"],
            )

            self.iou_scores.append(iou)

    def process_frame(
        self,
        frame,
        dt: float = 1.0 / 30.0,
    ):
        # For Carla we could use dt = current_timestamp - previous_timestamp
        detections, results = self.perception.detect_and_track(frame)

        self.evaluate_old_predictions(detections)

        detections = self.update_tracks_and_predict(
            detections,
            dt=dt,
        )

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