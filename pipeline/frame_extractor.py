"""
frame_extractor.py
──────────────────
Extracts a fixed number of evenly-sampled frames from a video file.
"""

import cv2


def extract_frames(video_path: str, max_frames: int = 30) -> list:
    """
    Open *video_path* and return up to *max_frames* BGR frames sampled
    evenly across the full duration.

    Returns
    -------
    list[np.ndarray]  – list of BGR frames (may be fewer than max_frames
                         if the video is short).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 10_000  # fallback for streams without frame count

    step = max(1, total // max_frames)
    frames: list = []

    frame_idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += step

    cap.release()
    print(f"[Frame Extractor] Extracted {len(frames)} frames from '{video_path}'")
    return frames
