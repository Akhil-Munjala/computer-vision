"""
keypoint_matcher.py
───────────────────
Stage 3 – SIFT Keypoint Detection & FLANN-based Matching.

Picks two frames from the middle of the video, detects SIFT features in
each, applies Lowe's ratio test, and draws the top matches side-by-side.
"""

import os
import cv2
import numpy as np


def match_keypoints(frames: list, session_id: str, output_folder: str) -> str:
    """
    Compute SIFT features on two video frames, match with FLANN, draw result.

    Returns
    -------
    str – path to the saved match visualisation JPEG.
    """
    if not frames:
        raise ValueError("No frames provided for keypoint matching.")

    if len(frames) == 1:
        # Single frame fallback – draw keypoints only
        return _keypoints_only(frames[0], session_id, output_folder)

    # ── Choose two frames well-separated in time ──────────────────────────
    mid   = len(frames) // 2
    gap   = max(1, len(frames) // 6)       # ~17 % of video apart
    idx1  = max(0, mid - gap)
    idx2  = min(len(frames) - 1, mid + gap)

    f1 = frames[idx1]
    f2 = frames[idx2]

    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # ── SIFT ──────────────────────────────────────────────────────────────
    sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.03)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    print(f"[Matcher] SIFT: {len(kp1)} kp in frame {idx1}, {len(kp2)} kp in frame {idx2}")

    if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
        return _keypoints_only(f1, session_id, output_folder)

    # ── FLANN matching ────────────────────────────────────────────────────
    index_params  = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=60)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    raw = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.72 * n.distance:
                good.append(m)

    print(f"[Matcher] Good matches after ratio test: {len(good)}")

    # ── Draw matches ──────────────────────────────────────────────────────
    draw_params = dict(
        matchColor=(57, 255, 20),           # neon green lines
        singlePointColor=(255, 100, 0),     # orange unmatched kp
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    vis = cv2.drawMatches(f1, kp1, f2, kp2, good[:60], None, **draw_params)

    # ── Overlay stats ─────────────────────────────────────────────────────
    _draw_match_hud(vis, len(kp1), len(kp2), len(good))

    out_path = os.path.join(output_folder, f"{session_id}_matches.jpg")
    cv2.imwrite(out_path, vis)
    print(f"[Matcher] Saved → {out_path}")
    return out_path


# ─── helpers ─────────────────────────────────────────────────────────────────

def _keypoints_only(frame, session_id, output_folder):
    """Draw SIFT keypoints on a single frame (fallback)."""
    sift = cv2.SIFT_create(nfeatures=500)
    kp, _ = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    vis = cv2.drawKeypoints(frame, kp, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(vis, f"SIFT Keypoints: {len(kp)}", (10, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (57, 255, 20), 2)
    out_path = os.path.join(output_folder, f"{session_id}_matches.jpg")
    cv2.imwrite(out_path, vis)
    return out_path


def _draw_match_hud(img, n1, n2, n_good):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 46), (12, 12, 22), -1)
    info = (f"SIFT  |  Frame A: {n1} kp   Frame B: {n2} kp   "
            f"FLANN matches: {n_good}  (ratio test 0.72)")
    cv2.putText(img, info, (10, 29),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (57, 255, 20), 1)
