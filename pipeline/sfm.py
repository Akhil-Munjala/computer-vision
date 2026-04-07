"""
sfm.py
──────
Stage 4 – Structure from Motion (SfM).

Pipeline:
  1. SIFT feature detection on N frames.
  2. Sequential pairwise matching with optional RANSAC homography pre-filter.
  3. Essential-matrix estimation → camera pose recovery (R, t).
  4. Linear triangulation → 3-D point cloud.
  5. Depth-map generation via per-pixel disparity + colormap.
"""

import os
import cv2
import numpy as np


def run_sfm(frames: list, session_id: str, output_folder: str):
    """
    Run SfM over *frames*.

    Returns
    -------
    (points_3d: np.ndarray | None, depth_map_path: str)
    """
    if len(frames) < 2:
        depth_path = _pseudo_depth(frames[0], session_id, output_folder)
        return None, depth_path

    # ── Camera intrinsics (estimated) ─────────────────────────────────────
    h, w = frames[0].shape[:2]
    f = max(w, h) * 1.05          # rough focal-length guess
    K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
                  [0, 0,     1]], dtype=np.float64)

    sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.025)

    # ── Collect keypoint descriptors for all frames ───────────────────────
    gray_list, kp_list, des_list = [], [], []
    for frm in frames:
        g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(g, None)
        gray_list.append(g)
        kp_list.append(kp)
        des_list.append(des)

    # ── Find the best pair (most good matches) ────────────────────────────
    best_pair = _find_best_pair(kp_list, des_list, max_search=min(len(frames) - 1, 8))
    if best_pair is None:
        depth_path = _pseudo_depth(frames[len(frames) // 2], session_id, output_folder)
        return None, depth_path

    i, j, pts1, pts2 = best_pair
    print(f"[SfM] Best pair: frames {i} ↔ {j}  ({len(pts1)} correspondences)")

    # ── Essential matrix + pose ───────────────────────────────────────────
    E, mask_E = cv2.findEssentialMat(pts1, pts2, K,
                                     method=cv2.RANSAC,
                                     prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        depth_path = _pseudo_depth(frames[i], session_id, output_folder)
        return None, depth_path

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    # Inlier points only
    inlier = mask_pose.ravel() > 0
    pts1_in = pts1[inlier]
    pts2_in = pts2[inlier]
    print(f"[SfM] Inliers after pose recovery: {inlier.sum()}")

    # ── Triangulation ─────────────────────────────────────────────────────
    P1 = K @ np.hstack([np.eye(3),   np.zeros((3, 1))])
    P2 = K @ np.hstack([R,            t              ])

    pts4d = cv2.triangulatePoints(P1, P2, pts1_in.T.astype(np.float64),
                                          pts2_in.T.astype(np.float64))
    pts3d = (pts4d[:3] / (pts4d[3] + 1e-9)).T   # (N, 3)

    # Filter outliers (keep points within ±200 units)
    valid = (np.abs(pts3d[:, 2]) < 200) & (pts3d[:, 2] > 0)
    pts3d = pts3d[valid]
    print(f"[SfM] Triangulated points (filtered): {len(pts3d)}")

    # ── Depth map ─────────────────────────────────────────────────────────
    depth_path = _stereo_depth_map(frames[i], gray_list[i],
                                   gray_list[j], session_id, output_folder)

    return pts3d if len(pts3d) > 4 else None, depth_path


# ─── helpers ─────────────────────────────────────────────────────────────────

def _find_best_pair(kp_list, des_list, max_search: int = 6):
    """Return the (i, j, pts1, pts2) pair with the most good FLANN matches."""
    index_params  = dict(algorithm=1, trees=5)
    search_params = dict(checks=60)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    best, best_i, best_j = None, 0, 1

    for i in range(max_search):
        for j in range(i + 1, min(i + 4, len(des_list))):
            d1, d2 = des_list[i], des_list[j]
            if d1 is None or d2 is None or len(d1) < 10 or len(d2) < 10:
                continue
            try:
                raw = flann.knnMatch(d1.astype(np.float32),
                                     d2.astype(np.float32), k=2)
            except Exception:
                continue

            good = [m for pair in raw if len(pair) == 2
                    for m, n in [pair] if m.distance < 0.72 * n.distance]
            if len(good) > (best if best is not None else 0):
                best   = len(good)
                best_i = i
                best_j = j

    if best is None or best < 8:
        return None

    # Re-extract point coords for the best pair
    d1, d2 = des_list[best_i], des_list[best_j]
    raw = flann.knnMatch(d1.astype(np.float32), d2.astype(np.float32), k=2)
    good = [m for pair in raw if len(pair) == 2
            for m, n in [pair] if m.distance < 0.72 * n.distance]

    pts1 = np.float32([kp_list[best_i][m.queryIdx].pt for m in good])
    pts2 = np.float32([kp_list[best_j][m.trainIdx].pt for m in good])
    return best_i, best_j, pts1, pts2


def _stereo_depth_map(color_frame, gray1, gray2, session_id, output_folder):
    """
    Compute a semi-dense depth map with StereoBM (using two grayscale frames
    as a pseudo stereo pair).  Falls back to a Laplacian pseudo-depth if dims
    mismatch.
    """
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    if h1 != h2 or w1 != w2:
        gray2 = cv2.resize(gray2, (w1, h1))

    # Make widths divisible by 16 (StereoBM requirement)
    pad_w = (16 - w1 % 16) % 16
    if pad_w:
        gray1 = cv2.copyMakeBorder(gray1, 0, 0, 0, pad_w, cv2.BORDER_REFLECT)
        gray2 = cv2.copyMakeBorder(gray2, 0, 0, 0, pad_w, cv2.BORDER_REFLECT)

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(gray1, gray2)          # int16, range ×16
    disp_norm = cv2.normalize(disparity, None, 0, 255,
                              cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Trim padding
    disp_norm = disp_norm[:h1, :w1]

    depth_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

    # Annotate
    cv2.putText(depth_colored, "Depth Map  (SfM – StereoBM)",
                (10, 34), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)
    _add_colorbar(depth_colored)

    out = os.path.join(output_folder, f"{session_id}_depth.jpg")
    cv2.imwrite(out, depth_colored)
    print(f"[SfM] Depth map saved → {out}")
    return out


def _pseudo_depth(frame, session_id, output_folder):
    """Monocular pseudo-depth via Laplacian + vertical gradient blend."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    lap  = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    h, w = gray.shape
    grad = np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1)
    grad = np.tile(grad, w)

    blend = cv2.addWeighted(lap, 0.55, grad, 0.45, 0)
    depth_colored = cv2.applyColorMap(blend, cv2.COLORMAP_TURBO)

    cv2.putText(depth_colored, "Depth Map  (Monocular estimate)",
                (10, 34), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)
    _add_colorbar(depth_colored)

    out = os.path.join(output_folder, f"{session_id}_depth.jpg")
    cv2.imwrite(out, depth_colored)
    return out


def _add_colorbar(img):
    """Paste a 'Near → Far' color bar in the bottom-right corner."""
    h, w = img.shape[:2]
    bar_w, bar_h = 160, 16
    x0, y0 = w - bar_w - 12, h - bar_h - 12
    bar = np.linspace(0, 255, bar_w, dtype=np.uint8).reshape(1, -1)
    bar = np.tile(bar, (bar_h, 1))
    bar_color = cv2.applyColorMap(bar, cv2.COLORMAP_TURBO)
    img[y0: y0 + bar_h, x0: x0 + bar_w] = bar_color
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)
    cv2.putText(img, "Near", (x0,         y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    cv2.putText(img, "Far",  (x0 + bar_w - 24, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
