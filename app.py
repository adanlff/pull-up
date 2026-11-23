import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE_DIR, ".yolo_cfg"))

from ultralytics import YOLO

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
model = YOLO(MODEL_PATH)

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)
]


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def parse_bool(val, default=True):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(val, (int, float)):
        return bool(val)
    return default


def calc_angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-5 or norm_bc < 1e-5:
        return None
    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def analyze_pose_and_draw(frame, conf_thres=0.25, show_keypoints=True, show_skeleton=True):
    results = model.predict(frame, conf=conf_thres, verbose=False)
    res = results[0]

    annotated = frame.copy()
    detections = []

    h_img, w_img = frame.shape[:2]
    scale_factor = max(1.0, min(w_img, h_img) / 640.0)
    box_thick = max(2, int(round(2 * scale_factor)))
    line_outer = max(4, int(round(4 * scale_factor)))
    line_inner = max(2, int(round(2 * scale_factor)))
    kpt_outer = max(6, int(round(6 * scale_factor)))
    kpt_inner = max(4, int(round(4 * scale_factor)))
    font_scale = max(0.55, 0.55 * scale_factor)
    font_thick = max(2, int(round(2 * scale_factor)))

    # Primary blue in BGR (#1A56DB -> RGB(26, 86, 219) -> BGR(219, 86, 26))
    color_primary = (219, 86, 26)
    color_light = (255, 255, 255)

    boxes = res.boxes
    keypoints = res.keypoints

    if boxes is not None and len(boxes) > 0:
        xyxy_list = boxes.xyxy.cpu().numpy()
        cls_list = boxes.cls.cpu().numpy().astype(int)
        conf_list = boxes.conf.cpu().numpy()

        kpts_xy = keypoints.xy.cpu().numpy() if keypoints is not None else None
        kpts_conf = keypoints.conf.cpu().numpy() if (keypoints is not None and keypoints.conf is not None) else None

        for i in range(len(xyxy_list)):
            x1, y1, x2, y2 = xyxy_list[i].astype(int)
            cls_id = int(cls_list[i])
            label = model.names.get(cls_id, str(cls_id)).upper()
            conf = float(conf_list[i])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_primary, box_thick)

            conf_pct = int(conf * 100)
            tag = f"{label} {conf_pct}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            pad_x = int(round(8 * scale_factor))
            pad_y = int(round(6 * scale_factor))
            y_tag = max(y1, th + pad_y * 2)
            cv2.rectangle(annotated, (x1, y_tag - th - pad_y * 2), (x1 + tw + pad_x * 2, y_tag), color_primary, -1)
            cv2.putText(
                annotated,
                tag,
                (x1 + pad_x, y_tag - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color_light,
                font_thick,
                cv2.LINE_AA
            )

            elbow_angle = None
            chin_up = False
            visible_kpts = 0
            kpts_list = []

            if kpts_xy is not None and i < len(kpts_xy):
                pts = kpts_xy[i]
                p_conf = kpts_conf[i] if kpts_conf is not None else np.ones(len(pts))

                for k_idx, pt in enumerate(pts):
                    c_val = float(p_conf[k_idx])
                    px, py = float(pt[0]), float(pt[1])
                    is_valid = c_val > 0.25 and px > 0 and py > 0
                    if is_valid:
                        visible_kpts += 1
                    kpts_list.append({
                        "x": round(px, 2),
                        "y": round(py, 2),
                        "conf": round(c_val, 4),
                        "valid": is_valid
                    })

                if show_skeleton:
                    for a_idx, b_idx in SKELETON:
                        if a_idx < len(pts) and b_idx < len(pts):
                            if p_conf[a_idx] > 0.25 and p_conf[b_idx] > 0.25:
                                pa = (int(pts[a_idx][0]), int(pts[a_idx][1]))
                                pb = (int(pts[b_idx][0]), int(pts[b_idx][1]))
                                if pa[0] > 0 and pa[1] > 0 and pb[0] > 0 and pb[1] > 0:
                                    cv2.line(annotated, pa, pb, color_light, line_outer, cv2.LINE_AA)
                                    cv2.line(annotated, pa, pb, color_primary, line_inner, cv2.LINE_AA)

                if show_keypoints:
                    for k_idx, pt in enumerate(pts):
                        if p_conf[k_idx] > 0.25 and pt[0] > 0 and pt[1] > 0:
                            px, py = int(pt[0]), int(pt[1])
                            cv2.circle(annotated, (px, py), kpt_outer, color_light, -1, cv2.LINE_AA)
                            cv2.circle(annotated, (px, py), kpt_inner, color_primary, -1, cv2.LINE_AA)

                angles = []
                if len(pts) >= 11:
                    if p_conf[5] > 0.25 and p_conf[7] > 0.25 and p_conf[9] > 0.25:
                        ang_l = calc_angle(pts[5], pts[7], pts[9])
                        if ang_l is not None:
                            angles.append(ang_l)
                    if p_conf[6] > 0.25 and p_conf[8] > 0.25 and p_conf[10] > 0.25:
                        ang_r = calc_angle(pts[6], pts[8], pts[10])
                        if ang_r is not None:
                            angles.append(ang_r)
                if angles:
                    elbow_angle = round(sum(angles) / len(angles), 1)

                if len(pts) >= 11:
                    wrist_ys = [pts[w][1] for w in (9, 10) if p_conf[w] > 0.25 and pts[w][1] > 0]
                    shoulder_ys = [pts[s][1] for s in (5, 6) if p_conf[s] > 0.25 and pts[s][1] > 0]
                    if elbow_angle is not None and elbow_angle < 105:
                        chin_up = True
                    elif wrist_ys and shoulder_ys and (min(shoulder_ys) - min(wrist_ys)) < 55:
                        chin_up = True

            detections.append({
                "label": label.lower(),
                "confidence": round(conf, 4),
                "elbow_angle": elbow_angle,
                "keypoints_detected": int(visible_kpts),
                "keypoints_total": 17,
                "keypoints": kpts_list,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "is_up": chin_up
            })

    return annotated, detections


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True) or {}
    img_b64 = data.get("image", "")
    conf_thres = float(data.get("conf", 0.25))
    show_keypoints = parse_bool(data.get("show_keypoints", True), True)
    show_skeleton = parse_bool(data.get("show_skeleton", True), True)

    if not img_b64:
        return jsonify({"error": "Gambar kosong"}), 400

    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]

    try:
        raw = base64.b64decode(img_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "Format gambar tidak valid"}), 400

    if frame is None:
        return jsonify({"error": "Gagal membaca gambar"}), 400

    annotated, detections = analyze_pose_and_draw(
        frame,
        conf_thres=conf_thres,
        show_keypoints=show_keypoints,
        show_skeleton=show_skeleton
    )
    ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return jsonify({"error": "Gagal memproses hasil"}), 500

    out_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    primary = detections[0] if detections else None
    return jsonify({
        "image": out_b64,
        "detections": detections,
        "primary": primary,
        "show_keypoints": show_keypoints,
        "show_skeleton": show_skeleton
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
