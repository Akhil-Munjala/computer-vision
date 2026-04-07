"""
detector.py
───────────
Stage 2 – Pothole Detection.

Strategy (two-pass):
  1. YOLOv8n detects general objects (person, car, etc.) to show the model
     running in real-time.
  2. A custom OpenCV dark-region + contour pipeline detects actual pothole
     candidates on the road surface (lower ⅔ of frame).

Both results are overlaid on the best mid-video frame and saved.
"""

import os
import cv2
import numpy as np


def detect_potholes(frames: list, session_id: str, output_folder: str) -> str:
    """
    Run pothole detection on *frames* and write an annotated JPEG.

    Returns
    -------
    str – absolute path of the saved detection image.
    """
    if not frames:
        raise ValueError("No frames provided for detection.")

    # ── Pick best frame (sharpest in middle third) ───────────────────────────
    frame = _pick_sharpest(frames)
    result = frame.copy()
    h, w = frame.shape[:2]

    # ── Pass 1: YOLOv8n ──────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')          # auto-downloads on first run
        yolo_results = model(frame, verbose=False)
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_name = r.names[int(box.cls[0])]
                cv2.rectangle(result, (x1, y1), (x2, y2), (50, 220, 255), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(result, label, (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 255), 2)
        print("[Detector] YOLOv8n detection complete.")
    except Exception as e:
        print(f"[Detector] YOLO skipped: {e}")

    # ── Pass 2: Dark-region pothole candidates ───────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Adaptive threshold picks out dark patches on a lighter road surface
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51, C=10
    )

    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_y_start = h // 3          # only look in lower ⅔ (road area)
    pothole_idx = 0
    pothole_regions = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600 or area > w * h * 0.4:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if y + bh < roi_y_start:
            continue
        # Aspect-ratio filter: potholes are roughly compact
        aspect = bw / (bh + 1e-6)
        if aspect > 6 or aspect < 0.15:
            continue

        pothole_idx += 1
        pothole_regions.append((x, y, bw, bh))

        # Draw filled semi-transparent overlay
        overlay = result.copy()
        cv2.drawContours(overlay, [cnt], -1, (0, 60, 255), -1)
        cv2.addWeighted(overlay, 0.35, result, 0.65, 0, result)

        # Bounding box + label
        cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 30, 255), 2)
        label = f"Pothole #{pothole_idx}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(result, (x, y - lh - 10), (x + lw + 6, y), (0, 30, 255), -1)
        cv2.putText(result, label, (x + 3, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # ── HUD overlay ──────────────────────────────────────────────────────────
    _draw_hud(result, pothole_idx, w, h)

    out_path = os.path.join(output_folder, f"{session_id}_detection.jpg")
    cv2.imwrite(out_path, result)
    print(f"[Detector] Saved → {out_path}  ({pothole_idx} pothole(s) found)")
    return out_path


# ─── helpers ─────────────────────────────────────────────────────────────────

def _pick_sharpest(frames: list):
    """Return the sharpest frame from the middle third of the list."""
    mid = len(frames) // 2
    candidates = frames[max(0, mid - 3): mid + 4]
    best, best_score = candidates[0], -1
    for f in candidates:
        score = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if score > best_score:
            best, best_score = f, score
    return best


def _draw_hud(img, count, w, h):
    """Draw a status HUD banner at the top of the image."""
    banner_h = 46
    banner = img[:banner_h].copy()
    cv2.rectangle(img, (0, 0), (w, banner_h), (15, 15, 25), -1)
    cv2.addWeighted(img[:banner_h], 0.0, banner, 1.0, 0, img[:banner_h])
    cv2.rectangle(img, (0, 0), (w, banner_h), (15, 15, 25), -1)

    cv2.putText(img, f"SfM Pothole Analyzer  |  Detected: {count} pothole(s)",
                (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 200, 255), 1)
    cv2.putText(img, "YOLOv8n + Adaptive Contour",
                (w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 180), 1)
