#!/usr/bin/env python3
"""
Render 8-view RGB images for DexGraspNet 2.0 scenes.

Uses Open3D offscreen rendering to place colored 3D meshes in scenes
according to XML annotations, then renders from 8 camera viewpoints
distributed around each scene.

Usage:
    python scripts/render_dexgraspnet.py --max-scenes 50   # Quick test
    python scripts/render_dexgraspnet.py --max-scenes 370  # All scenes_0
    python scripts/render_dexgraspnet.py --max-viewpoints 8 # 8 views per scene

Output:
    data/dexgraspnet/rendered/scene_XXXX/view_YY.png  (448×448 RGB)
"""

import os
import sys
import argparse
import logging
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_scene_objects(annot_path: str, mesh_base_dir: str) -> list:
    """Parse XML annotation to get object meshes + poses.
    
    DexGraspNet 2.0 mesh mapping:
      XML obj_name: '015_peach.ply' → mesh_base/015/textured.obj (preferred)
                                    → mesh_base/015/nontextured.ply (fallback)
      XML obj_name: 'camel.ply'    → search by name in mesh_base/*/
    """
    objects = []
    try:
        tree = ET.parse(annot_path)
        root = tree.getroot()
        
        for obj_elem in root.findall('.//obj'):
            pos_elem = obj_elem.find('pos_in_world')
            ori_elem = obj_elem.find('ori_in_world')
            name_elem = obj_elem.find('obj_name')
            
            if pos_elem is None or ori_elem is None or name_elem is None:
                continue
            
            obj_name = name_elem.text.strip()  # e.g., '015_peach.ply'
            
            # Extract numeric ID from name (e.g., '015' from '015_peach.ply')
            mesh_path = _resolve_mesh_path(obj_name, mesh_base_dir)
            if mesh_path is None:
                continue
            
            # Parse position
            pos = np.array([float(x) for x in pos_elem.text.strip().split()], dtype=np.float64)
            
            # Parse quaternion [qx, qy, qz, qw]
            ori = np.array([float(x) for x in ori_elem.text.strip().split()], dtype=np.float64)
            
            objects.append({
                'mesh_path': mesh_path,
                'position': pos,
                'orientation': ori,
                'name': obj_name,
            })
    except Exception as e:
        logger.warning(f"Failed to parse {annot_path}: {e}")
    
    return objects


def _resolve_mesh_path(obj_name: str, mesh_base: str) -> Optional[str]:
    """Resolve mesh file from DexGraspNet object name.
    
    '015_peach.ply' → meshdata/015/textured.obj or nontextured.ply
    'camel.ply'     → search meshdata/*/
    """
    name_no_ext = obj_name.replace('.ply', '').replace('.obj', '')
    parts = name_no_ext.split('_')
    
    # Try numeric prefix as directory ID
    if parts[0].isdigit():
        obj_id = parts[0]
        obj_dir = os.path.join(mesh_base, obj_id)
        if os.path.isdir(obj_dir):
            # Prefer textured, then nontextured
            for candidate in ['textured.obj', 'nontextured.ply', 'nontextured_simplified.ply']:
                path = os.path.join(obj_dir, candidate)
                if os.path.exists(path):
                    return path
    
    # Search all subdirectories for matching name
    base = Path(mesh_base)
    if base.exists():
        for sub in base.iterdir():
            if sub.is_dir():
                for candidate in ['textured.obj', 'nontextured.ply']:
                    path = sub / candidate
                    if path.exists():
                        return str(path)
    
    return None


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def generate_camera_viewpoints(center: np.ndarray, n_views: int = 8,
                                radius: float = 0.35) -> list:
    """Generate n_views camera eye positions around a scene center.
    
    DexGraspNet objects are typically at Z ≈ 0.45 on a table.
    Returns list of (eye, center, up) tuples for Open3D setup_camera(fov, center, eye, up).
    
    Views: 6 around circle at varying elevations + 1 high-angle + 1 top-down.
    """
    viewpoints = []
    n_circle = n_views - 2  # Reserve 2 for high-angle and top-down
    
    for i in range(n_circle):
        angle = 2 * np.pi * i / n_circle
        # Vary elevation: 25°-45° above horizontal
        elev_deg = 25 + 20 * np.sin(np.pi * i / n_circle)
        elev = np.radians(elev_deg)
        
        eye = center + np.array([
            radius * np.cos(angle) * np.cos(elev),
            radius * np.sin(angle) * np.cos(elev),
            radius * np.sin(elev),
        ])
        viewpoints.append((eye, center, np.array([0.0, 0.0, 1.0])))
    
    # High-angle view (60° elevation)
    eye = center + np.array([radius * 0.3, radius * 0.2, radius * 0.85])
    viewpoints.append((eye, center, np.array([0.0, 0.0, 1.0])))
    
    # Top-down view
    eye = center + np.array([0.01, 0.0, radius * 1.2])
    viewpoints.append((eye, center, np.array([0.0, 1.0, 0.0])))  # up = Y for top-down
    
    return viewpoints


def render_scene_open3d(objects: list, viewpoints: list,
                        image_size: int = 448) -> list:
    """Render scene from multiple viewpoints using Open3D offscreen renderer.
    
    Args:
        objects: List of dicts with mesh_path, position, orientation, name.
        viewpoints: List of (eye, center, up) tuples from generate_camera_viewpoints.
        image_size: Output image size (square).
    
    Returns:
        List of [H, W, 3] uint8 numpy arrays.
    """
    import open3d as o3d
    
    # Create renderer
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(image_size, image_size)
    except Exception:
        logger.warning("Open3D OffscreenRenderer not available")
        return _render_legacy_from_objects(objects, viewpoints, image_size)
    
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultLit'
    
    # Add table
    table = o3d.geometry.TriangleMesh.create_box(width=0.8, height=0.6, depth=0.005)
    table.translate([-0.4, -0.3, 0.375])
    table.paint_uniform_color([0.65, 0.55, 0.45])
    table.compute_vertex_normals()
    renderer.scene.add_geometry('table', table, mat)
    
    # Load and place objects
    n_loaded = 0
    for obj_info in objects:
        try:
            mesh_path = obj_info['mesh_path']
            mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
            if len(mesh.vertices) == 0:
                continue
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            if not mesh.has_vertex_colors():
                h = hash(obj_info['name']) % 1000
                mesh.paint_uniform_color([
                    max(0.2, (h * 37 % 256) / 255.0),
                    max(0.2, (h * 91 % 256) / 255.0),
                    max(0.2, (h * 159 % 256) / 255.0),
                ])
            
            R = quaternion_to_rotation_matrix(obj_info['orientation'])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = obj_info['position']
            mesh.transform(T)
            
            renderer.scene.add_geometry(f'obj_{n_loaded}', mesh, mat)
            n_loaded += 1
        except Exception as e:
            logger.debug(f"Mesh load failed: {obj_info.get('name')}: {e}")
    
    if n_loaded == 0:
        del renderer
        return []
    
    # Lighting
    renderer.scene.scene.set_sun_light([-0.5, -0.5, -1.0], [1.0, 1.0, 1.0], 60000)
    renderer.scene.scene.enable_sun_light(True)
    
    # Render from each viewpoint using look_at API
    rendered = []
    for (eye, center, up) in viewpoints:
        renderer.setup_camera(60.0, center.tolist(), eye.tolist(), up.tolist())
        img = renderer.render_to_image()
        rendered.append(np.asarray(img).copy())
    
    del renderer
    return rendered


def _render_legacy(geometries, viewpoints, image_size):
    """Fallback: render using point cloud projection (no GPU needed)."""
    import cv2
    
    rendered = []
    # Combine all mesh vertices into a point cloud
    all_points = []
    all_colors = []
    for geo in geometries:
        verts = np.asarray(geo.vertices)
        if geo.has_vertex_colors():
            colors = np.asarray(geo.vertex_colors)
        else:
            colors = np.ones_like(verts) * 0.5
        all_points.append(verts)
        all_colors.append(colors)
    
    if not all_points:
        return []
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    fx = fy = image_size * 1.2
    cx, cy = image_size / 2, image_size / 2
    
    for vp in viewpoints:
        # Project points
        pts_cam = (vp[:3, :3] @ all_points.T + vp[:3, 3:4]).T  # [N, 3]
        
        # Filter points in front of camera
        valid = pts_cam[:, 2] > 0.01
        pts_cam = pts_cam[valid]
        cols = all_colors[valid]
        
        if len(pts_cam) == 0:
            rendered.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))
            continue
        
        # Project to image
        u = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).astype(int)
        v = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).astype(int)
        
        # Z-buffer rendering
        img = np.zeros((image_size, image_size, 3), dtype=np.float32)
        zbuf = np.full((image_size, image_size), np.inf)
        
        mask = (u >= 0) & (u < image_size) & (v >= 0) & (v < image_size)
        u, v, z, c = u[mask], v[mask], pts_cam[:, 2][mask], cols[mask]
        
        # Sort by depth (far to near) so near objects overwrite far
        order = np.argsort(-z)
        u, v, z, c = u[order], v[order], z[order], c[order]
        
        for i in range(len(u)):
            if z[i] < zbuf[v[i], u[i]]:
                zbuf[v[i], u[i]] = z[i]
                img[v[i], u[i]] = c[i]
        
        # Fill holes with dilation
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        for ch in range(3):
            img_uint8[:, :, ch] = cv2.dilate(img_uint8[:, :, ch], kernel, iterations=2)
        
        rendered.append(img_uint8)
    
    return rendered


def render_all_scenes(data_dir: str, output_dir: str, n_views: int = 8,
                      image_size: int = 448, max_scenes: int = None):
    """Render all scenes and save RGB images."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scenes_dir = data_dir / 'scenes'
    mesh_base = str(data_dir / 'meshdata')
    
    if not scenes_dir.exists():
        logger.error(f"Scenes not found at {scenes_dir}")
        return
    
    # Find all scenes
    scene_dirs = sorted([d for d in scenes_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('scene_')])
    
    if max_scenes:
        scene_dirs = scene_dirs[:max_scenes]
    
    logger.info(f"Rendering {len(scene_dirs)} scenes × {n_views} views = {len(scene_dirs) * n_views} images")
    
    rendered_count = 0
    skipped_count = 0
    
    for i, scene_dir in enumerate(scene_dirs):
        scene_name = scene_dir.name
        scene_out = output_dir / scene_name
        
        # Skip if already rendered
        if scene_out.exists() and len(list(scene_out.glob('*.png'))) >= n_views:
            rendered_count += n_views
            continue
        
        # Find any annotation file to get object poses
        annot_dir = scene_dir / 'realsense' / 'annotations'
        if not annot_dir.exists():
            skipped_count += 1
            continue
        
        annot_files = sorted(annot_dir.glob('*.xml'))
        if not annot_files:
            skipped_count += 1
            continue
        
        # Use first annotation (objects are same across viewpoints, just different camera)
        objects = load_scene_objects(str(annot_files[0]), mesh_base)
        if not objects:
            skipped_count += 1
            continue
        
        # Compute scene center from object positions
        positions = np.array([o['position'] for o in objects])
        center = positions.mean(axis=0)
        
        # Generate camera viewpoints
        viewpoints = generate_camera_viewpoints(center, n_views=n_views)
        
        # Render
        try:
            images = render_scene_open3d(objects, viewpoints, image_size)
        except Exception as e:
            logger.warning(f"Render failed for {scene_name}: {e}")
            skipped_count += 1
            continue
        
        if not images:
            skipped_count += 1
            continue
        
        # Save
        scene_out.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        for v_idx, img_arr in enumerate(images):
            img_path = scene_out / f'view_{v_idx:02d}.png'
            Image.fromarray(img_arr).save(str(img_path))
        
        rendered_count += len(images)
        
        if (i + 1) % 20 == 0:
            logger.info(f"  {i+1}/{len(scene_dirs)} scenes, {rendered_count} images rendered, {skipped_count} skipped")
    
    logger.info(f"Done! {rendered_count} images rendered, {skipped_count} scenes skipped")
    logger.info(f"Output: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Render DexGraspNet 2.0 scenes')
    parser.add_argument('--data-dir', type=str, default='./data/dexgraspnet')
    parser.add_argument('--output-dir', type=str, default='./data/dexgraspnet/rendered')
    parser.add_argument('--n-views', type=int, default=8, help='Number of viewpoints per scene')
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--max-scenes', type=int, default=None, help='Limit number of scenes')
    args = parser.parse_args()
    
    render_all_scenes(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_views=args.n_views,
        image_size=args.image_size,
        max_scenes=args.max_scenes,
    )


if __name__ == '__main__':
    main()

