# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-JeffBoylan-Commercial
# © 2025 Jeff Boylan
"""
L/H Framework — Benchmark Log Analyzer
-------------------------------------
Reads CSV logs from 'logs/' directory and visualizes FPS & conservation
over time. Supports multiple files for comparison.
"""

import os, glob, csv
import numpy as np
import matplotlib.pyplot as plt

def read_log(path):
    t, scale, lam, fps, cons = [], [], [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row["timestamp"]))
            scale.append(float(row["scale"]))
            lam.append(float(row["lambda"]))
            fps.append(float(row["fps"]))
            cons.append(float(row["conservation"]))
    return np.array(t), np.array(scale), np.array(lam), np.array(fps), np.array(cons)

def summarize(label, fps, cons):
    print(f"\n📊 {label}")
    print(f"  FPS: mean={fps.mean():.2f}, std={fps.std():.2f}, min={fps.min():.2f}, max={fps.max():.2f}")
    print(f"  Cons: mean={cons.mean():.4f}, std={cons.std():.4f}, min={cons.min():.4f}, max={cons.max():.4f}")

def plot_logs(paths):
    plt.figure(figsize=(10,6))
    for p in paths:
        name = os.path.basename(p)
        t, s, l, fps, cons = read_log(p)
        t -= t[0]  # relative time
        plt.subplot(2,1,1)
        plt.plot(t, fps, label=f"FPS: {name}")
        plt.subplot(2,1,2)
        plt.plot(t, cons, label=f"Cons: {name}")
        summarize(name, fps, cons)

    plt.subplot(2,1,1)
    plt.title("L/H Benchmark Performance")
    plt.ylabel("FPS")
    plt.grid(True)
    plt.legend(fontsize=8)

    plt.subplot(2,1,2)
    plt.ylabel("Conservation")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    paths = sorted(glob.glob("logs/lh_benchmark_log_*.csv"))
    if not paths:
        print("No logs found in ./logs/")
    else:
        print(f"Found {len(paths)} log(s):")
        for p in paths: print(" -", p)
        plot_logs(paths)
