# risk_engine.py

driving_actions = ["go", "slow_down", "brake", "stop"]

# Action-risk thresholds
MIN_IOU = 0.03
STOP_IOU = 0.25

# Ego-zone danger threshold
DEFAULT_DANGER_IOU = 0.05


def make_ego_corridor(frame_w, frame_h):
    """
    Static danger zone in front of the ego vehicle, expressed in image pixels.
    """
    x1 = int(frame_w * 0.35)
    x2 = int(frame_w * 0.65)
    y1 = int(frame_h * 0.45)
    y2 = int(frame_h * 0.95)

    return [x1, y1, x2, y2]


def predict_ego_corridor(
    action,
    frame_w,
    frame_h,
    ego_speed_px_per_frame=8,
    horizon=5,
):
    """
    Approximate where the ego corridor will extend over the next few frames
    under a candidate longitudinal action.
    """
    ego_corridor = make_ego_corridor(frame_w, frame_h)

    if action == "go":
        shift = ego_speed_px_per_frame * horizon
    elif action == "slow_down":
        shift = ego_speed_px_per_frame * horizon * 0.4
    else:
        shift = 0

    x1, y1, x2, y2 = ego_corridor

    y1 = max(0, y1 - int(shift))
    y2 = max(0, y2 - int(shift))

    return [x1, y1, x2, y2]


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


def compute_action_risk(detections, frame_w, frame_h):
    """
    Scores candidate longitudinal actions by overlap between each action's
    predicted ego corridor and predicted object boxes.

    Returns:
        best_action: str
        action_risks: dict[str, float]
    """
    action_risks = {}
    max_overlap = 0.0

    for action in driving_actions:
        ego_future = predict_ego_corridor(action, frame_w, frame_h)
        max_risk = 0.0

        for det in detections:
            pred_box = det.get("pred_box")
            if pred_box is None:
                continue

            cy = (pred_box[1] + pred_box[3]) / 2

            # Ignore objects high in the image because they are still far away.
            if cy < frame_h * 0.45:
                continue

            iou = box_iou(ego_future, pred_box)
            max_overlap = max(max_overlap, iou)

            if iou < MIN_IOU:
                continue

            # Lower in the image = closer / more urgent.
            closeness = cy / frame_h
            risk = iou * closeness
            max_risk = max(max_risk, risk)

        action_risks[action] = max_risk

    if max_overlap >= STOP_IOU:
        return "stop", action_risks

    best_action = min(action_risks, key=action_risks.get)
    return best_action, action_risks


def compute_ego_danger(
    detections,
    frame_w,
    frame_h,
    danger_threshold=DEFAULT_DANGER_IOU,
):
    """
    Computes whether any predicted object box overlaps the ego danger zone.

    This is separate from action selection:
    - compute_action_risk() asks: which action looks safest?
    - compute_ego_danger() asks: is anything predicted to enter our danger zone?
    """
    ego_box = make_ego_corridor(frame_w, frame_h)

    max_ego_iou = 0.0
    danger_objects = []

    for det in detections:
        pred_box = det.get("pred_box")
        if pred_box is None:
            continue

        cy = (pred_box[1] + pred_box[3]) / 2

        # Keep the same "ignore far objects" rule used by action risk.
        if cy < frame_h * 0.45:
            continue

        ego_iou = box_iou(ego_box, pred_box)
        max_ego_iou = max(max_ego_iou, ego_iou)

        if ego_iou >= danger_threshold:
            danger_objects.append({
                "track_id": det.get("track_id"),
                "class": det.get("class"),
                "confidence": det.get("confidence"),
                "bbox": det.get("bbox"),
                "pred_box": pred_box,
                "ego_iou": ego_iou,
            })

    return {
        "ego_box": ego_box,
        "max_ego_iou": max_ego_iou,
        "danger": len(danger_objects) > 0,
        "danger_threshold": danger_threshold,
        "danger_objects": danger_objects,
    }