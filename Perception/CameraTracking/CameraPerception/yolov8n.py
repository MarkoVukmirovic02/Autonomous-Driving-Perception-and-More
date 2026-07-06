from ultralytics import YOLO


class YOLOPerception:
    def __init__(self, model_path, conf=0.25, imgsz=640, allowed_classes=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.allowed_classes = allowed_classes

    def detect_and_track(self, frame):
        results = self.model.track(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            persist=True,
            verbose=False,
        )[0]

        detections = []

        if results.boxes is None:
            return detections, results

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            if self.allowed_classes is not None and cls_name not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            detections.append({
                "track_id": track_id,
                "class": cls_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
            })

        return detections, results