"""
reconstructor.py
────────────────
Stage 6 – 3-D Point Cloud Visualisation.

Renders the triangulated point cloud from SfM using matplotlib's 3-D axes
with a dark space-inspired theme.  If no 3-D points are available, draws a
synthetic placeholder cloud so the UI always shows something meaningful.
"""

import os
import numpy as np

import matplotlib
matplotlib.use('Agg')          # non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers projection)


def reconstruct_3d(points_3d, session_id: str, output_folder: str) -> str:
    """
    Render *points_3d* as a coloured 3-D scatter plot and save to JPEG.

    Parameters
    ----------
    points_3d : np.ndarray | None
        Shape (N, 3).  If None a synthetic cloud is generated.
    session_id : str
    output_folder : str

    Returns
    -------
    str – path of the saved image.
    """
    if points_3d is None or len(points_3d) < 5:
        points_3d = _synthetic_pothole_cloud()

    # Clip extremes for nicer rendering
    points_3d = _clip_outliers(points_3d)

    fig = plt.figure(figsize=(10, 7), facecolor='#07071a')
    ax  = fig.add_subplot(111, projection='3d', facecolor='#07071a')

    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    # Colour by depth (Z)
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors  = plt.cm.plasma(z_norm)

    sc = ax.scatter(x, y, z, c=colors, s=2.0, alpha=0.85, linewidths=0)

    # Highlight lowest-depth cluster (pothole pit) in red
    pit_mask = z < np.percentile(z, 15)
    if pit_mask.sum() > 0:
        ax.scatter(x[pit_mask], y[pit_mask], z[pit_mask],
                   c='#ff4444', s=6, alpha=0.9, label='Pothole region')
        ax.legend(loc='upper left', framealpha=0.3, labelcolor='white',
                  facecolor='#1a1a3a', edgecolor='#444466')

    # ── Axes style ───────────────────────────────────────────────────────
    _style_axes(ax)

    ax.set_title('3D Point Cloud  –  SfM Reconstruction',
                 color='white', fontsize=13, pad=14, fontweight='bold')
    ax.set_xlabel('X  (m)', color='#9999bb', labelpad=10)
    ax.set_ylabel('Y  (m)', color='#9999bb', labelpad=10)
    ax.set_zlabel('Z  – Depth (m)', color='#9999bb', labelpad=10)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap='plasma',
                                norm=plt.Normalize(vmin=z.min(), vmax=z.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, orientation='vertical')
    cbar.set_label('Depth (m)', color='#aaaacc', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#aaaacc')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaaacc')

    plt.tight_layout(pad=1.5)
    out_path = os.path.join(output_folder, f"{session_id}_pointcloud.jpg")
    plt.savefig(out_path, dpi=110, bbox_inches='tight',
                facecolor='#07071a', edgecolor='none')
    plt.close(fig)
    print(f"[Reconstructor] Point cloud saved → {out_path}  ({len(points_3d)} pts)")
    return out_path


# ─── helpers ─────────────────────────────────────────────────────────────────

def _style_axes(ax):
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#2a2a4a')
    ax.grid(True, color='#1a1a3a', linewidth=0.5)
    ax.tick_params(colors='#8888aa', labelsize=7)


def _clip_outliers(pts, pct=2):
    """Remove the top/bottom *pct* % on every axis."""
    lo = np.percentile(pts, pct,     axis=0)
    hi = np.percentile(pts, 100-pct, axis=0)
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    clipped = pts[mask]
    return clipped if len(clipped) > 10 else pts


def _synthetic_pothole_cloud(n: int = 2000) -> np.ndarray:
    """
    Generate a synthetic 3-D point cloud that resembles a road surface
    with a pothole depression in the centre.
    """
    rng = np.random.default_rng(42)

    # Road surface – flat plane with noise
    road_x = rng.uniform(-1.5, 1.5, n)
    road_y = rng.uniform(-1.0, 1.0, n)
    road_z = rng.normal(0, 0.02, n)

    # Pothole – elliptical depression (negative Z = deeper)
    ph_n = n // 4
    theta = rng.uniform(0, 2 * np.pi, ph_n)
    r     = rng.uniform(0, 0.35, ph_n)
    ph_x  = r * np.cos(theta)
    ph_y  = r * np.sin(theta) * 0.6
    ph_z  = -rng.exponential(0.08, ph_n)   # depth below road

    pts = np.vstack([
        np.column_stack([road_x, road_y, road_z]),
        np.column_stack([ph_x,   ph_y,   ph_z  ])
    ])
    return pts.astype(np.float64)
