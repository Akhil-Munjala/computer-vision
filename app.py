import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from pipeline.frame_extractor import extract_frames
from pipeline.detector import detect_potholes
from pipeline.keypoint_matcher import match_keypoints
from pipeline.sfm import run_sfm
from pipeline.reconstructor import reconstruct_3d
from pipeline.estimator import estimate_dimensions

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = os.path.join('static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'mpg', 'mpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided.'}), 400

    video = request.files['video']
    if video.filename == '' or not allowed_file(video.filename):
        return jsonify({'error': 'Invalid or unsupported file type. Please upload a video file.'}), 400

    session_id = str(uuid.uuid4())[:8]
    filename = secure_filename(video.filename)
    video_path = os.path.join(UPLOAD_FOLDER, f'{session_id}_{filename}')
    video.save(video_path)

    try:
        # ── Stage 1: Frame Extraction ─────────────────────────────
        frames = extract_frames(video_path, max_frames=30)
        if not frames:
            return jsonify({'error': 'Could not extract frames from video.'}), 500

        # ── Stage 2: Pothole Detection (YOLO + CV) ────────────────
        detection_img_path = detect_potholes(frames, session_id, OUTPUT_FOLDER)

        # ── Stage 3: Keypoint Detection + Matching (SIFT/FLANN) ───
        match_img_path = match_keypoints(frames, session_id, OUTPUT_FOLDER)

        # ── Stage 4 & 5: SfM + Depth Map ─────────────────────────
        points_3d, depth_map_path = run_sfm(frames, session_id, OUTPUT_FOLDER)

        # ── Stage 6: 3D Point Cloud Visualization ────────────────
        pointcloud_path = reconstruct_3d(points_3d, session_id, OUTPUT_FOLDER)

        # ── Stage 7: Width & Depth Estimation ────────────────────
        metrics = estimate_dimensions(frames, points_3d)

        def to_url(path):
            return '/' + path.replace('\\', '/')

        return jsonify({
            'success': True,
            'session_id': session_id,
            'detection_image': to_url(detection_img_path),
            'match_image':     to_url(match_img_path),
            'depth_map':       to_url(depth_map_path),
            'pointcloud':      to_url(pointcloud_path),
            'metrics':         metrics
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up uploaded video to save disk space
        if os.path.exists(video_path):
            os.remove(video_path)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
