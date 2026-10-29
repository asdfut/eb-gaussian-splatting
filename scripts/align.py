"""Robust rigid transform mapping one model's camera poses onto another's (shared image names).

Usage: align.py <src_images.txt> <dst_images.txt> <out.npy> [--prefix h3/]
Fits T (dst_world <- src_world) on camera centers with RANSAC + Umeyama (no scale).
"""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("out")
ap.add_argument("--prefix", default="")
ap.add_argument("--thresh", type=float, default=0.3)
a = ap.parse_args()

def load(path):
    poses, lines = {}, [l for l in open(path) if not l.startswith("#")]
    for l in lines[::2] if path.endswith("images.txt") else lines:
        p = l.split()
        if len(p) < 10 or not p[9].startswith(a.prefix): continue
        qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
        Rcw = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4); T[:3, :3] = Rcw.T; T[:3, 3] = -Rcw.T @ [tx, ty, tz]
        poses[p[9]] = T
    return poses

def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((Y - my).T @ (X - mx))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    Rm = U @ D @ Vt; T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = my - Rm @ mx
    return T

src, dst = load(a.src), load(a.dst)
names = sorted(set(src) & set(dst))
X = np.array([src[n][:3, 3] for n in names]); Y = np.array([dst[n][:3, 3] for n in names])
rng = np.random.default_rng(0); best = None
for _ in range(2000):
    idx = rng.choice(len(names), 3, replace=False)
    T = umeyama(X[idx], Y[idx])
    inl = np.linalg.norm((X @ T[:3, :3].T + T[:3, 3]) - Y, axis=1) < a.thresh
    if best is None or inl.sum() > best.sum(): best = inl
T = umeyama(X[best], Y[best])
err = np.linalg.norm((X @ T[:3, :3].T + T[:3, 3]) - Y, axis=1)
rot_err = [np.degrees(R.from_matrix((T @ src[n])[:3, :3].T @ dst[n][:3, :3]).magnitude()) for n in names]
np.save(a.out, T)
print(f"common={len(names)} inliers={best.sum()} | center err median={np.median(err[best]):.3f}m "
      f"p90={np.percentile(err[best], 90):.3f}m | rot err median={np.median(np.array(rot_err)[best]):.2f}deg")
print("T =\n", np.round(T, 4))
