import cv2
import numpy as np

def traffic_light_color(crop):
    if crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red = red1 + red2

    yellow = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([95, 255, 255]))

    scores = {
        "red": cv2.countNonZero(red),
        "yellow": cv2.countNonZero(yellow),
        "green": cv2.countNonZero(green),
    }
    # reject dark red billboard-like regions
    v_channel = hsv[:, :, 2]
    bright_pixels = cv2.countNonZero(cv2.inRange(v_channel, 180, 255))

    if bright_pixels < 3:
        return "unknown"
    color = max(scores, key=scores.get)

    if scores[color] < 5:
        return "unknown"

    return color


def select_relevant_traffic_light(traffic_lights, frame_width, frame_height, maneuver="straight"):
    candidates = []

    for light in traffic_lights:
        x1, y1, x2, y2 = light["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)

        if cy > frame_height * 0.65:
            continue

        if maneuver == "left":
            target_x = frame_width * 0.35
        elif maneuver == "right":
            target_x = frame_width * 0.65
        else:
            target_x = frame_width * 0.50

        distance_to_target = abs(cx - target_x) / frame_width
        score = area * (1 - distance_to_target)

        candidates.append((score, light))

    if not candidates:
        return None

    return max(candidates, key=lambda x: x[0])[1]


def is_valid_traffic_light(det, frame_width, frame_height):
    x1, y1, x2, y2 = det["bbox"]
    w = x2 - x1
    h = y2 - y1
    area = w * h
    aspect = w / max(h, 1)
    cy = (y1 + y2) / 2

    if det["confidence"] < 0.25:
        return False

    if cy > frame_height * 0.55:
        return False

    if area > frame_width * frame_height * 0.005:
        return False

    if aspect > 1.5:
        return False

    return True
