# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-JeffBoylan-Commercial
# © 2025 Jeff Boylan
"""
L/H Framework — Real-time RGB Upscale Demo (Stable Edition)
---------------------------------------------------------------------
Controls:
  ↑ / ↓ : increase / decrease λ (lambda)
  ← / → : increase / decrease scale factor
  s     : save current frame
  q / ESC : quit
"""

import os
import re
import time
import csv
import collections
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import cv2


# ----------------------------- Helpers -----------------------------
def safe_filename(value: float, prefix: str = "") -> str:
    """Make a float safe for filenames (scientific → underscores)."""
    s = f"{value:.2e}"
    s = re.sub(r"[.e+-]", "_", s)
    return f"{prefix}{s}"


# ----------------------------- Core Ops -----------------------------
def gradient(A: np.ndarray):
    gx = 0.5 * (np.roll(A, -1, axis=1) - np.roll(A, 1, axis=1))
    gy = 0.5 * (np.roll(A, -1, axis=0) - np.roll(A, 1, axis=0))
    return gx, gy


def divergence(px: np.ndarray, py: np.ndarray):
    dx = 0.5 * (np.roll(px, -1, axis=1) - np.roll(px, 1, axis=1))
    dy = 0.5 * (np.roll(py, -1, axis=0) - np.roll(py, 1, axis=0))
    return dx + dy


def poisson_phi(A: np.ndarray, B: np.ndarray, lam: float = 1e-2):
    dI = B - A
    gx, gy = gradient(A)
    denom = gx * gx + gy * gy + lam
    return divergence(dI * gx, dI * gy) / np.maximum(denom, 1e-12)


def flow_and_warp_rgb(A: np.ndarray, phi: np.ndarray) -> np.ndarray:
    gx, gy = gradient(phi)
    H, W, _ = A.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y2 = np.clip((yy + gy).round().astype(int), 0, H - 1)
    x2 = np.clip((xx + gx).round().astype(int), 0, W - 1)
    return A[y2, x2, :]


# ----------------------------- Metrics -----------------------------
def conservation_score(A: np.ndarray, B: np.ndarray) -> float:
    """Relative mean intensity conservation."""
    return 1.0 - abs(A.mean() - B.mean())


# ----------------------------- Processing -----------------------------
MAX_DISPLAY_W, MAX_DISPLAY_H = 1920, 1080  # optional cap


def resize_for_display(img: Image.Image) -> Image.Image:
    if img.width > MAX_DISPLAY_W or img.height > MAX_DISPLAY_H:
        img = img.resize((MAX_DISPLAY_W, MAX_DISPLAY_H), Image.LANCZOS)
    return img


def process_and_combine(img: Image.Image, scale: float, lam: float):
    t0 = time.perf_counter()

    hi = np.asarray(img, dtype=np.float32) / 255.0
    lo = ImageOps.contain(img, (int(img.width / scale), int(img.height / scale)))
    lo_up = lo.resize(img.size, Image.BICUBIC)
    lo_np = np.asarray(lo_up, dtype=np.float32) / 255.0

    lum = 0.2989 * lo_np[..., 0] + 0.5870 * lo_np[..., 1] + 0.1140 * lo_np[..., 2]
    lum_hi = 0.2989 * hi[..., 0] + 0.5870 * hi[..., 1] + 0.1140 * hi[..., 2]

    phi = poisson_phi(lum, lum_hi, lam)
    lh_up = np.clip(flow_and_warp_rgb(lo_np, phi), 0, 1)

    combined = np.concatenate([hi, lo_np, lh_up], axis=1)
    combo_img = (combined * 255).astype(np.uint8)
    combo_img = np.ascontiguousarray(combo_img[..., ::-1])  # RGB → BGR, contiguous

    elapsed = time.perf_counter() - t0
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    score = conservation_score(lo_np, lh_up)

    return combo_img, fps, score


# ----------------------------- Graph Overlay -----------------------------
def draw_graph_bars(frame: np.ndarray, fps_hist, cons_hist, maxlen: int = 100):
    h, w = frame.shape[:2]
    graph_h = 60
    overlay = np.zeros((graph_h * 2, w, 3), dtype=np.uint8)

    # FPS (green)
    fps_max = max(60, max(fps_hist) if fps_hist else 60)
    for i, val in enumerate(fps_hist):
        x = int(i * w / maxlen)
        y = int(graph_h * (1 - min(val / fps_max, 1)))
        cv2.line(overlay, (x, graph_h - 1), (x, y), (0, 255, 0), 1)

    # Conservation (cyan-ish)
    for i, val in enumerate(cons_hist):
        x = int(i * w / maxlen)
        y = int(graph_h + graph_h * (1 - val))
        cv2.line(overlay, (x, 2 * graph_h - 1), (x, y), (255, 200, 0), 1)

    cv2.putText(overlay, f"FPS {fps_hist[-1]:.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(overlay, f"Cons {cons_hist[-1]:.3f}", (10, graph_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    y0 = h - overlay.shape[0]
    frame[y0:h, :] = cv2.addWeighted(frame[y0:h, :], 0.6, overlay, 0.4, 0)

    return frame


# ----------------------------- Logging -----------------------------
def init_csv_log() -> str:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"lh_demo_log_{int(time.time())}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "scale", "lambda", "fps", "conservation"])
    return str(path)


def append_csv_log(path: str, scale: float, lam: float, fps: float, cons: float):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([time.time(), scale, lam, fps, cons])


# ----------------------------- Main Loop -----------------------------
def lh_upscale_realtime(img_path: str | None = None, use_camera: bool = False):
    scale, lam = 2.0, 1e-2
    maxlen = 100
    fps_hist = collections.deque(maxlen=maxlen)
    cons_hist = collections.deque(maxlen=maxlen)
    log_path = init_csv_log()
    print(f"[LOG] Writing benchmark data to: {log_path}")

    # ---------- Input ----------
    capture = None
    static_img: Image.Image | None = None

    if use_camera:
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] Cannot open camera.")
            return
    else:
        if img_path is None or not Path(img_path).exists():
            print(f"[ERROR] Image not found: {img_path}")
            return
        try:
            static_img = Image.open(img_path).convert("RGB")
            static_img = resize_for_display(static_img)
        except Exception as e:
            print(f"[ERROR] Failed to load image: {e}")
            return

    # ---------- UI ----------
    win_name = "L/H Real-time Demo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print("Controls: ↑↓ λ | ←→ scale | s=save | q/ESC=quit")

    while True:
        # ---- Grab frame ----
        if use_camera:
            ret, frame_bgr = capture.read()
            if not ret:
                print("[WARN] Camera frame not received.")
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
        else:
            img = static_img

        # ---- Process ----
        combo_img, fps, score = process_and_combine(img, scale, lam)
        fps_hist.append(fps)
        cons_hist.append(score)
        append_csv_log(log_path, scale, lam, fps, score)

        # ---- Overlay info ----
        h, _ = combo_img.shape[:2]
        info = f"Scale={scale:.2f} λ={lam:.2e} FPS={fps:6.1f} Cons={score:.4f}"
        cv2.putText(combo_img, info, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        combo_img = draw_graph_bars(combo_img, fps_hist, cons_hist, maxlen)

        # Update window title
        cv2.setWindowTitle(win_name, f"L/H Demo — {info}")

        cv2.imshow(win_name, combo_img)

        # ---- Keyboard ----
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):          # ESC or q
            break
        elif key == ord('s'):              # save screenshot
            lam_str = safe_filename(lam, "lam")
            out_path = f"lh_s{scale:.1f}_{lam_str}_fps{fps:.1f}_cons{score:.4f}.png"
            cv2.imwrite(out_path, combo_img)
            print(f"[SAVED] {out_path}")
        elif key == 0:                     # Up arrow
            lam *= 1.5
        elif key == 1:                     # Down arrow
            lam /= 1.5
        elif key == 3:                     # Right arrow
            scale = min(scale + 0.25, 6.0)
        elif key == 2:                     # Left arrow
            scale = max(scale - 0.25, 1.25)

    # ---------- Cleanup ----------
    if capture:
        capture.release()
    cv2.destroyAllWindows()
    print(f"[DONE] CSV log saved to {log_path}")


# ----------------------------- Entry -----------------------------
if __name__ == "__main__":
    # Toggle use_camera=True for webcam
    lh_upscale_realtime("example_input.png", use_camera=False)