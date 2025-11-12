# -----------------------------------------------------------
# © 2025 Jeff Boylan
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-JeffBoylan-Commercial
# DOI: 10.5281/zenodo.17589521
# -----------------------------------------------------------

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import time, math
import matplotlib.pyplot as plt

# ----------------------------- Core 3D Ops -----------------------------
def gradient_3d(A):
    gx = 0.5 * (np.roll(A, -1, 2) - np.roll(A, 1, 2))
    gy = 0.5 * (np.roll(A, -1, 1) - np.roll(A, 1, 1))
    gz = 0.5 * (np.roll(A, -1, 0) - np.roll(A, 1, 0))
    return gx, gy, gz

def divergence_3d(px, py, pz):
    dx = 0.5 * (np.roll(px, -1, 2) - np.roll(px, 1, 2))
    dy = 0.5 * (np.roll(py, -1, 1) - np.roll(py, 1, 1))
    dz = 0.5 * (np.roll(pz, -1, 0) - np.roll(pz, 1, 0))
    return dx + dy + dz

def poisson_phi_3d(A, B, lam=1e-2):
    dI = B - A
    gx, gy, gz = gradient_3d(A)
    denom = gx*gx + gy*gy + gz*gz + lam
    return divergence_3d(dI*gx, dI*gy, dI*gz) / np.maximum(denom, 1e-12)

def warp_3d(A, phi):
    gx, gy, gz = gradient_3d(phi)
    D, H, W = A.shape
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    z2 = np.clip(zz + gz, 0, D-1)
    y2 = np.clip(yy + gy, 0, H-1)
    x2 = np.clip(xx + gx, 0, W-1)
    coords = np.vstack([z2.ravel(), y2.ravel(), x2.ravel()])
    return map_coordinates(A, coords, order=1, mode='nearest').reshape(D, H, W)

# ----------------------------- Metrics -----------------------------
def conservation_score_3d(A, B):
    return 1.0 - abs(A.mean() - B.mean())

def l2(A, B):
    return np.sqrt(np.mean((A - B)**2))

def mae(A, B):
    return np.mean(np.abs(A - B))

def psnr(A, B):
    mse = np.mean((A - B)**2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))

# ----------------------------- Torus generator -----------------------------
def torus_volume(D=64, H=64, W=64, center=None, R=16, r=6, intensity=1.0):
    if center is None:
        center = (D/2, H/2, W/2)
    z, y, x = np.mgrid[0:D, 0:H, 0:W]
    x, y, z = x-center[2], y-center[1], z-center[0]
    val = (np.sqrt(x**2 + y**2) - R)**2 + z**2
    mask = val <= r**2
    vol = np.zeros((D,H,W), np.float32)
    vol[mask] = intensity
    return gaussian_filter(vol, sigma=0.8)

def make_scene(D=64, H=64, W=64, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Randomly offset and scale torus
    R1, r1 = rng.uniform(14,18), rng.uniform(5,8)
    R2, r2 = rng.uniform(13,19), rng.uniform(4,8)
    cA = (D/2 + rng.integers(-3,3), H/2 + rng.integers(-3,3), W/2 + rng.integers(-3,3))
    cB = (D/2 + rng.integers(-6,6), H/2 + rng.integers(-6,6), W/2 + rng.integers(-6,6))
    A = torus_volume(D,H,W,cA,R1,r1)
    B = torus_volume(D,H,W,cB,R2,r2)
    return A,B

# ----------------------------- Visualization -----------------------------
def plot_slices(A, B, phi, warped):
    mid = A.shape[0]//2
    error = np.abs(B - warped)
    fig, axes = plt.subplots(1,5, figsize=(16,4))
    titles = ["A[zmid]", "B[zmid]", "φ[zmid]", "Warped[zmid]", "Error[zmid]"]
    for ax, data, title in zip(axes, [A,B,phi,warped,error], titles):
        im = ax.imshow(data[mid], cmap='viridis')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# ----------------------------- Circulation diagnostic -----------------------------
from scipy.ndimage import map_coordinates

def circulation_midloop(phi, r_frac=0.35, n=2048, slice_index=None):
    """∮ ∇φ · dl on a circular contour in the mid z-slice."""
    if slice_index is None:
        slice_index = phi.shape[0] // 2
    sl = phi[slice_index]
    gy, gx = np.gradient(sl)
    h, w = sl.shape
    cx, cy = w / 2.0, h / 2.0
    r = r_frac * min(cx, cy)

    theta = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    gx_s = map_coordinates(gx, [y, x], order=1, mode='nearest')
    gy_s = map_coordinates(gy, [y, x], order=1, mode='nearest')

    tx = -r * np.sin(theta)
    ty =  r * np.cos(theta)
    dtheta = 2.0 * np.pi / n

    circ = np.sum(gx_s * tx + gy_s * ty) * dtheta
    return circ

# ----------------------------- Trial -----------------------------
def run_trial(D=64, H=64, W=64, lam=1e-2, visualize=True):
    A,B = make_scene(D,H,W)
    t0 = time.perf_counter()
    phi = poisson_phi_3d(A,B,lam)
    warped = warp_3d(A,phi)
    t = time.perf_counter()-t0

    if visualize:
        plot_slices(A,B,phi,warped)

    cons = conservation_score_3d(A,warped)
    p = psnr(B,warped)
    circ = circulation_midloop(phi)

    print(f"Conservation={cons:.6f}, PSNR={p:.2f} dB, Time={t:.3f}s")
    print(f"Circulation integral (∮∇φ·dl) = {circ:.6e}")

# ----------------------------- Entry -----------------------------
if __name__ == "__main__":
    run_trial(D=64, H=64, W=64, lam=1e-2)
