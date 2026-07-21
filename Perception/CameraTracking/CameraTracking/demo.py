import time
import cv2
from CameraTracking.tracker_manager import PerceptionRiskModule

from risk.risk_engine import (
    make_ego_corridor,
    predict_ego_corridor,
)

from CameraPerception.traffic_light import (
    traffic_light_color,
    select_relevant_traffic_light,
    is_valid_traffic_light,
)

latencies = []

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
video_path = BASE_DIR / "data_video" / "5.mp4"

cap = cv2.VideoCapture(str(video_path))


module = PerceptionRiskModule(
    model_path="yolov8n.pt",
    conf=0.10,
    imgsz=960,
    predict_ahead=5,
)


while True:

    frame_start = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        break

    output = module.process_frame(frame)

    detections = output["detections"]
    results = output["results"]
    best_action = output["best_action"]
    action_risks = output["action_risks"]
    avg_iou = output["avg_prediction_iou"]

    frame_h, frame_w = frame.shape[:2]

    print(f"Action: {best_action}, risks: {action_risks}", flush=True)


    ego_go = predict_ego_corridor("go", frame_w, frame_h)
    ego_slow = predict_ego_corridor("slow_down", frame_w, frame_h)
    ego_stop = predict_ego_corridor("stop", frame_w, frame_h)



    print(
        f"Ego future corridors | go={ego_go} slow={ego_slow} stop={ego_stop}",
        flush=True
    )

    
      
    
    for det in detections:
        if det["class"] == "traffic light":
            x1, y1, x2, y2 = det["bbox"]
            crop = frame[y1:y2, x1:x2]
            det["state"] = traffic_light_color(crop)
            
    
    annotated = frame.copy()

    ego_now = make_ego_corridor(frame_w, frame_h)
    ego_future = predict_ego_corridor(best_action, frame_w, frame_h)

    cv2.rectangle(annotated, (ego_now[0], ego_now[1]), (ego_now[2], ego_now[3]), (0, 255, 255), 2)
    cv2.putText(annotated, "ego_now", (ego_now[0], ego_now[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.rectangle(annotated, (ego_future[0], ego_future[1]), (ego_future[2], ego_future[3]), (255, 0, 255), 2)
    cv2.putText(annotated, f"ego_{best_action}_t+5", (ego_future[0], ego_future[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.putText(annotated, f"ACTION: {best_action}", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)



    traffic_lights = [
        det for det in detections
        if det["class"] == "traffic light"
        and det.get("state") != "unknown"
        and is_valid_traffic_light(det, frame.shape[1], frame.shape[0])
    ]

    planned_maneuver = "straight"

    relevant_light = select_relevant_traffic_light(
        traffic_lights,
        frame.shape[1],
        frame.shape[0],
        maneuver=planned_maneuver,
    )

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        if det["class"] == "traffic light":
            if det.get("state") == "unknown":
                continue

            if not is_valid_traffic_light(det, frame.shape[1], frame.shape[0]):
                continue

            label = f"traffic light: {det['state']} {conf:.2f}"

            if relevant_light is not None and det is relevant_light:
                label = f"RELEVANT: {det['state']} {conf:.2f}"

        else:
            label = f"{det['class']} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        if "pred_box" in det:
            px1, py1, px2, py2 = det["pred_box"]

            cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 0, 0), 2)

            cv2.putText(
                annotated,
                "pred",
                (px1, max(20, py1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

    frame_end = time.perf_counter()
    latency_ms = (frame_end - frame_start) * 1000
    latencies.append(latency_ms)
    cv2.rectangle(annotated, (ego_go[0], ego_go[1]), (ego_go[2], ego_go[3]), (255, 0, 255), 2)
    cv2.putText(annotated, "ego_go_t+5", (ego_go[0], ego_go[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(
        annotated,
        f"Latency: {latency_ms:.1f} ms",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    print(f"Frame {module.frame_idx}: {latency_ms:.2f} ms")

    cv2.imshow("Perception", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if len(latencies) > 0:
    avg_latency = sum(latencies) / len(latencies)
    print("Average latency:", avg_latency, "ms")
    print("Approx FPS:", 1000 / avg_latency)

if len(module.iou_scores) > 0:
    avg_iou = sum(module.iou_scores) / len(module.iou_scores)
    print("Average IoU:", avg_iou)

cap.release()
cv2.destroyAllWindows()

#server_client.send_hazard(vehicle_id, risk, action)


#objects, lights, signs = perception.process_frame(frame)

#v2x_messages = server_client.get_messages(vehicle_id)

#risk, action = risk_engine.compute(
#    objects=objects,
#    lights=lights,
#   signs=signs,
#    v2x_messages=v2x_messages,
#)

#server_client.send_hazard(vehicle_id, risk, action)

