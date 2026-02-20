#!/usr/bin/env python3
"""
Visualize an Episode HDF5 file as a video with task instruction.

Usage:
    python scripts/visualize_episode.py data/episodes/taco_play/ep_000000.hdf5
    python scripts/visualize_episode.py data/episodes/aloha/ep_000000.hdf5 --fps 10
    python scripts/visualize_episode.py data/episodes/bc_z/ep_000005.hdf5 -o bc_z_ep5.mp4
"""

import argparse
import h5py
import numpy as np
import cv2
from pathlib import Path


def visualize_episode(hdf5_path: str, output_path: str = None, fps: int = 15):
    """Render an episode HDF5 as a side-by-side multi-view video."""

    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        print(f"ERROR: {hdf5_path} not found")
        return

    with h5py.File(str(hdf5_path), 'r') as f:
        # ── Metadata ──
        ds_name = f.attrs.get('dataset_name', 'unknown')
        ep_len = int(f.attrs.get('episode_length', 0))
        n_views = int(f.attrs.get('n_views', 0))
        img_size = int(f.attrs.get('image_size', 448))

        # ── Language ──
        lang = ""
        if 'language' in f:
            val = f['language'][()]
            if isinstance(val, bytes):
                lang = val.decode('utf-8')
            else:
                lang = str(val)
        if not lang:
            lang = "(no instruction)"

        # ── Actions ──
        actions = f['actions'][:] if 'actions' in f else None  # [T, 128]
        action_mask = f['action_mask'][:] if 'action_mask' in f else None  # [128]

        # ── Load all views ──
        views = []
        view_names = []
        for v in range(n_views):
            key = f'images/view_{v}'
            if key in f:
                imgs = f[key][:]  # [T, H, W, 3] uint8
                views.append(imgs)
                view_names.append(f'view_{v}')

        if not views:
            print(f"WARNING: No image views found in {hdf5_path}")
            # Create placeholder
            views = [np.zeros((max(ep_len, 1), 224, 224, 3), dtype=np.uint8)]
            view_names = ['(no images)']

    T = views[0].shape[0]
    n_v = len(views)

    # ── Layout: grid of views + info bar ──
    # 1-3 views: single row    |  4-6: 2 rows of 3  |  7-8: 2 rows of 4
    target_h, target_w = 256, 256
    if n_v <= 3:
        grid_cols = n_v
        grid_rows = 1
    elif n_v <= 6:
        grid_cols = 3
        grid_rows = 2
    else:
        grid_cols = 4
        grid_rows = 2

    bar_h = 64  # info bar height
    min_canvas_w = 600  # minimum width so text is always readable
    canvas_w = max(target_w * grid_cols, min_canvas_w)
    canvas_h = target_h * grid_rows + bar_h

    # ── Output path ──
    if output_path is None:
        output_path = str(hdf5_path.with_suffix('.mp4'))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    print(f"Episode: {hdf5_path.name}")
    print(f"  Dataset: {ds_name}")
    print(f"  Task: {lang}")
    print(f"  Frames: {T}, Views: {n_v} ({', '.join(view_names)})")
    if n_v > 3:
        print(f"  Layout: {grid_rows}×{grid_cols} grid")
    if actions is not None and action_mask is not None:
        n_active = int(action_mask.sum())
        print(f"  Actions: [{T}, 128] ({n_active} active dims)")
    print(f"  Output: {output_path}")

    # Adaptive font scale
    font_scale = max(0.35, min(0.6, canvas_w / 900))

    for t in range(T):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # ── Info bar ──
        canvas[:bar_h, :] = 30  # dark gray
        margin = 10
        usable_w = canvas_w - 2 * margin

        # Line 1: dataset + task (pixel-precise truncation)
        task_text = f"[{ds_name}] {lang}"
        fs = font_scale
        while True:
            (tw, _), _ = cv2.getTextSize(task_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            if tw <= usable_w or len(task_text) <= 10:
                break
            task_text = task_text[:-4] + "..."  # chop 1 char + re-add "..."
        cv2.putText(canvas, task_text, (margin, 24),
                     cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)

        # Line 2: frame info + action summary
        info = f"Frame {t+1}/{T}"
        if actions is not None and action_mask is not None:
            active = action_mask.astype(bool)
            act_vals = actions[t][active]
            info += f"  |  Action: mean={act_vals.mean():.3f} std={act_vals.std():.3f}"
        fs2 = font_scale * 0.85
        (tw2, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, fs2, 1)
        if tw2 > usable_w:
            info = f"Frame {t+1}/{T}  |  Act: m={act_vals.mean():.2f} s={act_vals.std():.2f}"
        cv2.putText(canvas, info, (margin, 52),
                     cv2.FONT_HERSHEY_SIMPLEX, fs2, (180, 220, 255), 1)

        # ── View tiles (grid layout) ──
        for v_idx, (view_imgs, vname) in enumerate(zip(views, view_names)):
            img = view_imgs[t]  # [H, W, 3]
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = cv2.resize(img, (target_w, target_h))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            row = v_idx // grid_cols
            col = v_idx % grid_cols
            y_off = bar_h + row * target_h
            x_off = col * target_w
            canvas[y_off:y_off + target_h, x_off:x_off + target_w] = img_bgr

            # View label
            cv2.putText(canvas, vname, (x_off + 5, y_off + 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        writer.write(canvas)

    writer.release()
    print(f"  ✅ Saved: {output_path} ({T} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize an Episode HDF5 as video with task instruction')
    parser.add_argument('hdf5_path', type=str, help='Path to episode HDF5 file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output video path (default: same name .mp4)')
    parser.add_argument('--fps', type=int, default=15, help='Video FPS')
    args = parser.parse_args()

    visualize_episode(args.hdf5_path, args.output, args.fps)


if __name__ == '__main__':
    main()

