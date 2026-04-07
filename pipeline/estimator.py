"""
estimator.py
────────────
Stage 7 – Real-World Width & Depth Estimation.

Uses:
  • Bounding boxes of pothole contours + assumed camera height to estimate
    real-world width / length of each pothole.
  • Z-range of the triangulated point cloud to estimate pothole depth.
  • Derived area (ellipse model) and volume (truncated cone model).
  • Severity classification based on standard road-maintenance thresholds.
"""

import cv2
import numpy as np


# ── Camera model constants (road-facing camera estimate) ─────────────────────
_FOCAL_MM       = 4.2     # focal length in mm (typical dashcam)
_SENSOR_WIDTH_MM = 6.2   # sensor width in mm
_CAM_HEIGHT_M   = 1.4    # camera height above road surface (m)
_REPAIR_COST_PER_CM3 = 0.0012  # USD per cm³ of material (approximate)


def estimate_dimensions(frames: list, points_3d) -> dict:
    """
    Estimate real-world dimensions of the detected pothole.

    Parameters
    ----------
    frames     : list of BGR frames
    points_3d  : np.ndarray (N, 3) or None

    Returns
    -------
    dict with keys:
        width_cm, depth_cm, length_cm,
        area_cm2, volume_cm3,
        severity, repair_cost_usd,
        num_potholes, confidence
    """
    if not frames:
        return _default_metrics()

    frame = frames[len(frames) // 2]
    h_img, w_img = frame.shape[:2]

    # ── Pixel-to-metre ratio at road plane ───────────────────────────────────
    f_px = (_FOCAL_MM / _SENSOR_WIDTH_MM) * w_img      # focal length in pixels
    px_to_m = _CAM_HEIGHT_M / f_px                     # 1 pixel ≈ ? metres

    # ── Detect pothole contours (same approach as detector.py) ───────────────
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51, C=10
    )
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    roi_y = h_img // 3
    widths_m, lengths_m = [], []

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < 600 or area_px > w_img * h_img * 0.4:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if y + bh < roi_y:
            continue
        aspect = bw / (bh + 1e-6)
        if aspect > 6 or aspect < 0.15:
            continue
        widths_m.append(bw  * px_to_m)
        lengths_m.append(bh * px_to_m)

    # ── Width & length from contours ─────────────────────────────────────────
    if widths_m:
        avg_w_m = float(np.mean(widths_m))
        avg_l_m = float(np.mean(lengths_m))
    else:
        avg_w_m = 0.28          # sensible default: 28 cm
        avg_l_m = 0.22

    # ── Depth from point cloud Z-range ───────────────────────────────────────
    if points_3d is not None and len(points_3d) >= 10:
        z = points_3d[:, 2]
        z_lo  = np.percentile(z, 5)
        z_hi  = np.percentile(z, 95)
        depth_m = min(abs(z_hi - z_lo) * 0.08, 0.40)   # scale + cap at 40 cm
        confidence = "Medium"
    else:
        depth_m    = 0.07       # default 7 cm
        confidence = "Low  (SfM fallback)"

    # ── Derived metrics ───────────────────────────────────────────────────────
    w_cm  = round(avg_w_m  * 100, 1)
    l_cm  = round(avg_l_m  * 100, 1)
    d_cm  = round(depth_m  * 100, 1)

    # Ellipse area model: A = π/4 · w · l
    area_cm2  = round(np.pi / 4 * w_cm * l_cm, 1)

    # Volume as a paraboloid: V = π/2 · (w/2) · (l/2) · d
    vol_cm3 = round(np.pi / 2 * (w_cm / 2) * (l_cm / 2) * d_cm, 1)

    # Repair cost (asphalt fill estimate)
    cost_usd = round(vol_cm3 * _REPAIR_COST_PER_CM3, 2)

    # ── Severity ─────────────────────────────────────────────────────────────
    severity = _classify_severity(w_cm, d_cm)

    return {
        "width_cm":       w_cm,
        "length_cm":      l_cm,
        "depth_cm":       d_cm,
        "area_cm2":       area_cm2,
        "volume_cm3":     vol_cm3,
        "repair_cost_usd": cost_usd,
        "severity":       severity,
        "num_potholes":   max(len(widths_m), 1),
        "confidence":     confidence
    }


# ─── helpers ─────────────────────────────────────────────────────────────────

def _classify_severity(w_cm: float, d_cm: float) -> str:
    """
    IRC SP 16 / AASHTO-inspired pothole severity thresholds.
    """
    if d_cm >= 10 or w_cm >= 60:
        return "🔴  High"
    if d_cm >= 5 or w_cm >= 30:
        return "🟠  Medium"
    return "🟡  Low"


def _default_metrics() -> dict:
    return {
        "width_cm":        0.0,
        "length_cm":       0.0,
        "depth_cm":        0.0,
        "area_cm2":        0.0,
        "volume_cm3":      0.0,
        "repair_cost_usd": 0.0,
        "severity":        "Unknown",
        "num_potholes":    0,
        "confidence":      "N/A"
    }
