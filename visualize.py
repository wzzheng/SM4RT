# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
# Source: https://github.com/ByteDance-Seed/TraceAnything
# Modified by Shing Ho J. Lin on April 2026: Modified viser program for visualization of our method. 


import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import os
import time
import threading
import argparse
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch.nn.functional as F

import numpy as np
import torch
import cv2
import viser

# ---- tiny helpers ----
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# ---- tiny HSV colormap (no heavy deps) ----
import colorsys
def hsv_colormap(vals01: np.ndarray) -> np.ndarray:
    """vals01: [T] in [0,1] -> [T,3] RGB in [0,1]"""
    vals01 = np.clip(vals01, 0.0, 1.0)
    rgb = [colorsys.hsv_to_rgb(v, 1.0, 1.0) for v in vals01]
    return np.asarray(rgb, dtype=np.float32)

# ----------------------- I/O -----------------------
def load_world_points_data(rgb_path) -> Dict:
    """Load world_points tensor and RGB images"""
    pts_path = f"./output/inference_wp_track.pt"
    world_points = torch.load(pts_path)
    B, T, H, W, _ = world_points.shape
    from PIL import Image
    img_pil = Image.open(rgb_path).convert('RGB')
    rgb_np = np.array(img_pil).astype(np.float32) / 255.0  # [H, W, 3] in [0, 1]
    rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    rgb_tensor = torch.nn.functional.interpolate(rgb_tensor, size=(H, W), mode='bilinear', align_corners=False)

    return {
        "world_points": world_points,
        "rgb": rgb_tensor,
        "num_frames": world_points.shape[1],
        "height": world_points.shape[2],
        "width": world_points.shape[3]
    }

# -------------- precompute tensors for viewer -------------
def build_precomputes_simple(
    data: Dict,
    t_step: float,
    ds: int,
) -> Tuple[np.ndarray, List[Dict]]:
    world_points = data["world_points"]  # [1, 8, H, W, 3]
    rgb_data = data["rgb"]  # [1, 3, H, W]
    num_frames = data["num_frames"]
    H_orig, W_orig = data["height"], data["width"]
    
    # timeline - use frame indices as time
    t_vals = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    T = len(t_vals)

    frames: List[Dict] = []
    stride = slice(None, None, ds)  # ::ds

    rgb_img = to_numpy(rgb_data[0].permute(1, 2, 0))  # [H, W, 3]
    
    if rgb_img.min() < 0:
        rgb_img_normalized = (rgb_img + 1.0) * 0.5
    else:
        rgb_img_normalized = rgb_img.copy()
    
    rgb_img_uint8 = np.clip(rgb_img_normalized * 255, 0, 255).astype(np.uint8)
    rgb_img_uint8 = rgb_img_uint8[::ds, ::ds]  # downsample
    H, W = rgb_img_uint8.shape[:2]
    img_flat = (rgb_img_uint8.astype(np.float32) / 255.0).reshape(-1, 3)

    for i in range(num_frames):
        print(f"Processing frame {i+1}/{num_frames}")
        
        frame_points = to_numpy(world_points[0, i])  # [518, 518, 3]
        
        if ds > 1:
            frame_points = frame_points[::ds, ::ds, :]
        
        points_flat = frame_points.reshape(-1, 3)
        
        if points_flat.shape[0] == img_flat.shape[0]:
            colors_flat = img_flat
        else:
            print(f"Warning: point count {points_flat.shape[0]} != color count {img_flat.shape[0]}")
            orig_coords = np.array(np.meshgrid(np.arange(H_orig//ds), np.arange(W_orig//ds))).T.reshape(-1, 2)
            new_coords = np.array(np.meshgrid(np.arange(H), np.arange(W))).T.reshape(-1, 2)
            scale_y = (H_orig//ds - 1) / max(1, H - 1)
            scale_x = (W_orig//ds - 1) / max(1, W - 1)
            scaled_coords = new_coords * [scale_y, scale_x]
            scaled_coords = np.clip(scaled_coords.astype(int), 0, [H_orig//ds-1, W_orig//ds-1])
            colors_flat = img_flat[scaled_coords[:, 0] * (W_orig//ds) + scaled_coords[:, 1]]
        
        frames.append(dict(
            img_rgb_uint8=rgb_img_uint8,
            points=points_flat,
            colors=colors_flat,
            H=H, W=W,
            frame_idx=i
        ))

    return t_vals, frames

# ---------------- nodes & state ----------------
class SimpleViewerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_updating = False
        self.status_label = None
        self.slider_time = None
        self.point_size = None
        self.frame_nodes = []  # len num_frames
        # trajectories
        self.show_traj = None
        self.traj_width = None
        self.traj_frames_text = None
        self.traj_build_btn = None
        self.traj_nodes = []
        self.playing = False
        self.play_btn = None
        self.pause_btn = None
        self.fps_slider = None
        self.loop_checkbox = None

def build_frame_nodes(server: viser.ViserServer, frames: List[Dict], state: SimpleViewerState):
    print("\n[viewer] Building frame nodes...")
    for i, fr in enumerate(frames):
        node = server.scene.add_point_cloud(
            name=f"/frame/{i:02d}",
            points=fr["points"],
            colors=fr["colors"],
            point_size=state.point_size.value if state.point_size else 0.0002,
            point_shape="rounded",
            visible=(i == 0),
        )
        state.frame_nodes.append(node)
        print(f"[viewer] Frame {i:02d}: added {fr['points'].shape[0]} points")

def _update_traj_visibility(state: "SimpleViewerState", server: viser.ViserServer, tidx: int, on: bool):
    if not state.traj_nodes:
        return
    with server.atomic():
        for t, nodes in enumerate(state.traj_nodes):
            vis = on and (t <= tidx)
            for nd in nodes:
                nd.visible = vis
    server.flush()

# ---------------- trajectories ----------------
def build_traj_nodes_simple(
    server: viser.ViserServer,
    frames: List[Dict],
    traj_frames: List[int],
    max_points: int,
    state: "SimpleViewerState",
):
    # remove old trajectories
    for lst in state.traj_nodes:
        for nd in lst:
            nd.remove()
    
    num_frames = len(frames)
    state.traj_nodes = [[] for _ in range(num_frames - 1)]

    vals01 = (np.arange(num_frames - 1, dtype=np.float32)) / max(1, num_frames - 2)
    seg_colors = hsv_colormap(vals01)  # [T-1, 3]

    print("[traj] building trajectories...")
    
    for fi in traj_frames:
        if fi < 0 or fi >= len(frames) - 1:
            continue
        
        current_frame = frames[fi]
        next_frame = frames[fi + 1]
        
        points_current = current_frame["points"]  # [N, 3]
        points_next = next_frame["points"]  # [N, 3]
        
        min_points = min(points_current.shape[0], points_next.shape[0])
        if min_points == 0:
            continue
        
        if max_points > 0 and min_points > max_points:
            indices = np.random.choice(min_points, size=max_points, replace=False)
            points_current = points_current[indices]
            points_next = points_next[indices]
            min_points = max_points
        
        segments = np.stack([points_current[:min_points], points_next[:min_points]], axis=1)  # [N, 2, 3]
        
        segment_color = seg_colors[fi]
        colors = np.tile(segment_color, (min_points, 1))  # [N, 3]
        
        node = server.scene.add_line_segments(
            name=f"/traj/frame{fi}_to_{fi+1}",
            points=segments,
            colors=np.repeat(colors[:, None, :], 2, axis=1),  # [N, 2, 3]
            line_width=state.traj_width.value if state.traj_width else 0.075,
            visible=False,
        )
        state.traj_nodes[fi].append(node)
        print(f"[traj] Added {min_points} trajectories from frame {fi} to {fi+1}")
    
    print("[traj] done.")

# ----------------- main viewer -----------------
def serve_simple_view(data: Dict, port: int = 8020, t_step: float = 0.125, ds: int = 2):
    server = viser.ViserServer(port=port)
    server.gui.set_panel_label("Simple Point Cloud Viewer")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False)

    # restore camera & scene setup
    server.scene.set_up_direction((0.0, -1.0, 0.0))
    server.scene.world_axes.visible = False

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle):
        with client.atomic():
            client.camera.position = (0.0, 0.0, 3.0)
            client.camera.look_at((0.0, 0.0, 0.0))
        client.flush()

    t_vals, frames = build_precomputes_simple(data, t_step, ds)

    state = SimpleViewerState()
    with server.gui.add_folder("Point Size", expand_by_default=True):
        state.point_size = server.gui.add_slider("Point Size", min=1e-5, max=1e-1, step=1e-4, initial_value=0.002)

    with server.gui.add_folder("Playback", expand_by_default=True):
        state.slider_time = server.gui.add_slider("Time", min=0.0, max=1.0, step=t_step, initial_value=0.0)
        state.play_btn = server.gui.add_button("▶ Play")
        state.pause_btn = server.gui.add_button("⏸ Pause")
        state.fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=2)
        state.loop_checkbox = server.gui.add_checkbox("Loop", True)

    # ---- Trajectories panel ----
    with server.gui.add_folder("Trajectories", expand_by_default=True):
        state.show_traj = server.gui.add_checkbox("Show trajectories", False)
        state.traj_width = server.gui.add_slider("Line width", min=0.01, max=0.2, step=0.005, initial_value=0.075)
        state.traj_frames_text = server.gui.add_text("Frames (e.g. 0,mid,last)", initial_value="0,mid,last")
        state.traj_build_btn = server.gui.add_button("Build / Refresh")

    state.status_label = server.gui.add_markdown("")

    # build nodes
    build_frame_nodes(server, frames, state)
    print("\n[viewer] Ready. Open the printed URL and play with sliders!\n")

    # --- playback loop thread ---
    def _playback_loop():
        while True:
            if state.playing:
                try:
                    fps = max(1, int(state.fps_slider.value)) if state.fps_slider else 10
                    dt = 1.0 / float(fps)
                    tv = float(state.slider_time.value)
                    tv_next = tv + t_step
                    if tv_next > 1.0 - 1e-6:
                        if state.loop_checkbox and state.loop_checkbox.value:
                            tv_next = 0.0
                        else:
                            tv_next = 1.0 - 1e-6
                            state.playing = False
                    state.slider_time.value = tv_next
                except Exception:
                    pass
            time.sleep(dt if state.playing else 0.05)

    threading.Thread(target=_playback_loop, daemon=True).start()

    # callbacks
    @state.slider_time.on_update
    def _(_):
        tv = state.slider_time.value
        tidx = int(round(tv / t_step))
        tidx = max(0, min(tidx, len(t_vals) - 1))
        with server.atomic():
            for t in range(len(state.frame_nodes)):
                state.frame_nodes[t].visible = (t == tidx)
        server.flush()
        # Update trajectory visibility
        _update_traj_visibility(state, server, tidx, on=(state.show_traj and state.show_traj.value))

    @state.play_btn.on_click
    def _(_):
        state.playing = True

    @state.pause_btn.on_click
    def _(_):
        state.playing = False

    @state.point_size.on_update
    def _(_):
        with server.atomic():
            for n in state.frame_nodes:
                n.point_size = state.point_size.value
        server.flush()

    # --- trajectories build/refresh and live controls ---
    def _parse_traj_frames():
        txt = (state.traj_frames_text.value or "").strip()
        if not txt:
            return []
        tokens = [t.strip() for t in txt.split(",") if t.strip()]
        n = len(frames)
        out_idx = []
        for tk in tokens:
            if tk == "mid":
                out_idx.append(n // 2)
            elif tk == "last":
                out_idx.append(n - 1)
            else:
                try:
                    out_idx.append(int(tk))
                except Exception:
                    pass
        out_idx = [i for i in sorted(set(out_idx)) if 0 <= i < n]
        return out_idx

    @state.traj_build_btn.on_click
    def _(_):
        sel = _parse_traj_frames()
        if not sel:
            return
        if state.status_label:
            state.status_label.value = "🧵 Building trajectories…"
        build_traj_nodes_simple(
            server=server,
            frames=frames,
            traj_frames=sel,
            max_points=5000,  # 减少最大点数以提高性能
            state=state,
        )
        tidx = int(round(state.slider_time.value / t_step))
        tidx = max(0, min(tidx, len(t_vals) - 1))
        _update_traj_visibility(state, server, tidx, on=(state.show_traj and state.show_traj.value))
        if state.status_label:
            state.status_label.value = ""

    @state.show_traj.on_update
    def _(_):
        tidx = int(round(state.slider_time.value / t_step))
        tidx = max(0, min(tidx, len(t_vals) - 1))
        _update_traj_visibility(state, server, tidx, on=state.show_traj.value)

    @state.traj_width.on_update
    def _(_):
        w = state.traj_width.value
        with server.atomic():
            for nodes in state.traj_nodes:
                for nd in nodes:
                    nd.line_width = w
        server.flush()

    return server

def parse_args():
    p = argparse.ArgumentParser("Simple Point Cloud Viewer")
    p.add_argument("--port", type=int, default=8200)
    p.add_argument("--t_step", type=float, default=1/30)
    p.add_argument("--ds", type=int, default=4, help="downsample stride for H,W (>=1)")
    p.add_argument("--rgb", type=str, default="./assets/swing/00000.jpg", help="downsample stride for H,W (>=1)")
    return p.parse_args()

def main():
    args = parse_args()
    data = load_world_points_data(args.rgb)
    server = serve_simple_view(data, port=args.port, t_step=args.t_step, ds=max(1, args.ds))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
    