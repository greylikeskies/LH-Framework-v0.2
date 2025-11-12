# -----------------------------------------------------------
# © 2025 Jeff Boylan
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-JeffBoylan-Commercial
# DOI: 10.5281/zenodo.17589521
# -----------------------------------------------------------
"""
L/H Framework — 3D Volume Laplacian Visualization Test
------------------------------------------------------
Extends the 2D harmonic alignment into 3D volumes.
Computes Δφ = ∇·(ΔI·g)/(‖g‖²+λ), warps A→B, and measures
3D conservation. Includes both 2D slice visualization and
3D isosurface rendering.

Run inside Jupyter or standalone.

Dependencies:
    pip install numpy scipy matplotlib pyvista
"""

import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import os, time
import pyvista as pv

# ----------------------------- Core 3D Ops -----------------------------
def gradient_3d(A):
    gx = 0.5 * (np.roll(A, -1, 2) - np.roll(A, 1, 2))  # ∂/∂x
    gy = 0.5 * (np.roll(A, -1, 1) - np.roll(A, 1, 1))  # ∂/∂y
    gz = 0.5 * (np.roll(A, -1, 0) - np.roll(A, 1, 0))  # ∂/∂z
    return gx, gy, gz

def divergence_3d(px, py, pz):
    dx = 0.5 * (np.roll(px, -1, 2) - np.roll(px, 1, 2))
    dy = 0.5 * (np.roll(py, -1, 1) - np.roll(py, 1, 1))
    dz = 0.5 * (np.roll(pz, -1, 0) - np.roll(pz, 1, 0))
    return dx + dy + dz

def poisson_phi_3d(A, B, lam=1e-2):
    """3D Laplacian potential field."""
    dI = B - A
    gx, gy, gz = gradient_3d(A)
    denom = gx*gx + gy*gy + gz*gz + lam
    return divergence_3d(dI*gx, dI*gy, dI*gz) / np.maximum(denom, 1e-12)

def warp_3d(A, phi):
    """Trilinear warp of A using ∇φ as displacement."""
    gx, gy, gz = gradient_3d(phi)
    D, H, W = A.shape
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    z2 = np.clip(zz + gz, 0, D-1)
    y2 = np.clip(yy + gy, 0, H-1)
    x2 = np.clip(xx + gx, 0, W-1)
    coords = np.vstack([z2.ravel(), y2.ravel(), x2.ravel()])
    return map_coordinates(A, coords, order=1, mode='nearest').reshape(D, H, W)

def conservation_score_3d(A, B):
    return 1.0 - abs(A.mean() - B.mean())

# ----------------------------- Volume Generator -----------------------------
def generate_blob(D=64, H=64, W=64, offset=(0,0,0)):
    z, y, x = np.mgrid[0:D, 0:H, 0:W]
    c = np.array([D/2+offset[0], H/2+offset[1], W/2+offset[2]])
    r = np.sqrt((x-c[2])**2 + (y-c[1])**2 + (z-c[0])**2)
    return np.exp(-r**2 / (2*(D/8)**2))

# ----------------------------- Visualization -----------------------------
def visualize_3d(phi, warped=None):
    """Interactive 3D isosurface viewer using PyVista."""
    D, H, W = phi.shape
    grid = pv.UniformGrid()
    grid.dimensions = np.array(phi.shape) + 1
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    grid.point_data["phi"] = phi.flatten(order="F")

    pl = pv.Plotter()
    pl.add_volume(phi, cmap="viridis", opacity="sigmoid_5")
    pl.add_isosurface(grid, scalars="phi", opacity=0.25, color="orange")

    if warped is not None:
        gx, gy, gz = gradient_3d(phi)
        skip = (slice(None,None,4),)*3
        pts = np.column_stack(np.nonzero(np.ones_like(phi[skip])))
        arrows = np.column_stack([gx[skip].ravel(), gy[skip].ravel(), gz[skip].ravel()])
        vec = pv.Arrow(start=pts.mean(axis=0), direction=arrows.mean(axis=0))
        pl.add_mesh(vec, color="red")

    pl.add_axes()
    pl.show_grid()
    pl.show()

# ----------------------------- Experiment -----------------------------
def run_volume_test(D=64, H=64, W=64, lam=1e-2, plot_slices=True, render_3d=False, save=True):
    print(f"Running 3D L/H volume test ({D}×{H}×{W}) λ={lam}")
    A = generate_blob(D, H, W, offset=(0,0,0))
    B = generate_blob(D, H, W, offset=(4,2,-3))  # shifted target blob

    t0 = time.perf_counter()
    phi = poisson_phi_3d(A, B, lam)
    warped = warp_3d(A, phi)
    elapsed = time.perf_counter() - t0

    C = conservation_score_3d(A, warped)
    print(f"Conservation = {C:.6f}, time = {elapsed:.3f}s")

    if plot_slices:
        mid = D//2
        fig, ax = plt.subplots(2,3, figsize=(10,7))
        ax[0,0].imshow(A[mid,:,:], cmap='magma');  ax[0,0].set_title("A[zmid]")
        ax[0,1].imshow(B[mid,:,:], cmap='magma');  ax[0,1].set_title("B[zmid]")
        ax[0,2].imshow(phi[mid,:,:], cmap='inferno'); ax[0,2].set_title("φ[zmid]")
        ax[1,0].imshow(warped[mid,:,:], cmap='magma'); ax[1,0].set_title("Warped[zmid]")
        ax[1,1].imshow((B-warped)[mid,:,:], cmap='coolwarm'); ax[1,1].set_title("Error[zmid]")
        ax[1,2].plot(np.mean(A, axis=(1,2)), label='A mean')
        ax[1,2].plot(np.mean(warped, axis=(1,2)), label='Warped mean')
        ax[1,2].legend(); ax[1,2].set_title("Mean Intensity per z")
        plt.tight_layout(); plt.show()

    if render_3d:
        visualize_3d(phi, warped)

    if save:
        os.makedirs("vol3d_out", exist_ok=True)
        np.save("vol3d_out/A.npy", A)
        np.save("vol3d_out/B.npy", B)
        np.save("vol3d_out/phi.npy", phi)
        np.save("vol3d_out/warped.npy", warped)
        print("Saved volumes to vol3d_out/")

    return C, elapsed

# ----------------------------- Entry -----------------------------
if __name__ == "__main__":
    run_volume_test(D=64, H=64, W=64, lam=1e-2, plot_slices=True, render_3d=True)
