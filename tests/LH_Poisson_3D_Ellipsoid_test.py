# -----------------------------------------------------------
# © 2025 Jeff Boylan
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-JeffBoylan-Commercial
# DOI: 10.5281/zenodo.17589521
# -----------------------------------------------------------

#L/H Poisson Test On Ellipsoid 3D Objects


import numpy as np
from scipy.ndimage import map_coordinates
import time, os, math

# ----------------------------- Core 3D L/H ops -----------------------------
def gradient_3d(A):
    gx = 0.5 * (np.roll(A, -1, 2) - np.roll(A, 1, 2))  # x
    gy = 0.5 * (np.roll(A, -1, 1) - np.roll(A, 1, 1))  # y
    gz = 0.5 * (np.roll(A, -1, 0) - np.roll(A, 1, 0))  # z
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
    # intensity conservation proxy
    return 1.0 - abs(A.mean() - B.mean())

def l2(A, B):
    return np.sqrt(np.mean((A - B)**2))

def mae(A, B):
    return np.mean(np.abs(A - B))

def psnr(A, B):
    mse = np.mean((A - B)**2)
    if mse <= 1e-12: 
        return float("inf")
    # assume volumes normalized to [0,1]
    return 20.0 * math.log10(1.0 / math.sqrt(mse))

# ----------------------------- Ellipsoid generator -----------------------------
def rotation_matrix_xyz(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

def ellipsoid_volume(D=64, H=64, W=64, center=None, radii=(12,9,6), euler=(0,0,0), intensity=1.0):
    z, y, x = np.mgrid[0:D, 0:H, 0:W]
    if center is None:
        center = (D/2, H/2, W/2)
    # shift to center
    X = np.stack([x-center[2], y-center[1], z-center[0]], axis=0)  # (3,D,H,W)
    R = rotation_matrix_xyz(*euler)  # (3,3)
    XR = R @ X.reshape(3, -1)        # rotate
    xr, yr, zr = XR[0], XR[1], XR[2]
    # ellipsoid implicit function
    val = (xr/radii[2])**2 + (yr/radii[1])**2 + (zr/radii[0])**2
    vol = np.exp(-np.clip(val, 0, 16)) * intensity  # soft ellipsoid blob
    return vol.reshape(D, H, W).astype(np.float32)

def make_scene(D=64, H=64, W=64, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Randomized A (one or two ellipsoids)
    A = np.zeros((D,H,W), np.float32)
    kA = rng.integers(1,3)
    for _ in range(kA):
        center = (D/2 + rng.integers(-6,6), H/2 + rng.integers(-6,6), W/2 + rng.integers(-6,6))
        radii  = (rng.integers(6,14), rng.integers(5,12), rng.integers(4,10))
        euler  = (rng.uniform(0, np.pi), rng.uniform(0, np.pi), rng.uniform(0, np.pi))
        A += ellipsoid_volume(D,H,W, center, radii, euler, intensity=rng.uniform(0.7,1.0))
    A = np.clip(A, 0, 1)

    # Randomized B (independent transform)
    B = np.zeros((D,H,W), np.float32)
    kB = kA  # keep same count to avoid trivial amplitude mismatch
    for _ in range(kB):
        center = (D/2 + rng.integers(-10,10), H/2 + rng.integers(-10,10), W/2 + rng.integers(-10,10))
        radii  = (rng.integers(6,14), rng.integers(5,12), rng.integers(4,10))
        euler  = (rng.uniform(0, np.pi), rng.uniform(0, np.pi), rng.uniform(0, np.pi))
        B += ellipsoid_volume(D,H,W, center, radii, euler, intensity=rng.uniform(0.7,1.0))
    B = np.clip(B, 0, 1)

    return A, B

# ----------------------------- Single trial -----------------------------
def run_trial(D=64, H=64, W=64, lam=1e-2, save=False, outdir="vol3d_non_spherical", seed=None):
    rng = np.random.default_rng(seed)
    A, B = make_scene(D,H,W, rng=rng)

    t0 = time.perf_counter()
    phi = poisson_phi_3d(A, B, lam)
    warped = warp_3d(A, phi)
    elapsed = time.perf_counter() - t0

    C  = conservation_score_3d(A, warped)
    l2e = l2(B, warped)
    mae_e = mae(B, warped)
    psnr_v = psnr(B, warped)

    if save:
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, "A.npy"), A)
        np.save(os.path.join(outdir, "B.npy"), B)
        np.save(os.path.join(outdir, "phi.npy"), phi)
        np.save(os.path.join(outdir, "warped.npy"), warped)

    return dict(conservation=C, l2=l2e, mae=mae_e, psnr=psnr_v, time=elapsed)

# ----------------------------- Battery -----------------------------
def run_battery(trials=5, D=64, H=64, W=64, lam=1e-2, save_first=False):
    stats = []
    for i in range(trials):
        r = run_trial(D,H,W, lam, save=(save_first and i==0), seed=i*9973)
        stats.append(r)
        print(f"[{i+1}/{trials}] "
              f"C={r['conservation']:.6f}  L2={r['l2']:.5f}  MAE={r['mae']:.5f}  "
              f"PSNR={r['psnr']:.2f} dB  t={r['time']:.3f}s")

    # aggregate
    def agg(key): 
        arr = np.array([s[key] for s in stats])
        return arr.mean(), arr.std(), arr.min(), arr.max()

    c_mu, c_sd, c_min, c_max   = agg('conservation')
    l2_mu, l2_sd, _, _         = agg('l2')
    mae_mu, mae_sd, _, _       = agg('mae')
    psnr_mu, psnr_sd, _, _     = agg('psnr')
    t_mu, t_sd, _, _           = agg('time')

    print("\n=== Battery Summary ===")
    print(f"Conservation: mean={c_mu:.6f}  sd={c_sd:.6f}  min={c_min:.6f}  max={c_max:.6f}")
    print(f"L2 error   : mean={l2_mu:.6f}  sd={l2_sd:.6f}")
    print(f"MAE        : mean={mae_mu:.6f}  sd={mae_sd:.6f}")
    print(f"PSNR       : mean={psnr_mu:.2f} dB  sd={psnr_sd:.2f} dB")
    print(f"Runtime    : mean={t_mu:.3f}s  sd={t_sd:.3f}s")

if __name__ == "__main__":
    # quick sanity run; bump trials/size as needed
    run_battery(trials=5, D=64, H=64, W=64, lam=1e-2, save_first=True)

