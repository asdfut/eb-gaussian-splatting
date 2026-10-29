"""Cross-session image pairs from a COLMAP text model: for each image of --b, the nearest
--k images of --a within --dist metres whose viewing directions differ by < 60 deg."""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("images_txt"); ap.add_argument("out")
ap.add_argument("--a", default="h1/"); ap.add_argument("--b", default="h3/")
ap.add_argument("--dist", type=float, default=3.0); ap.add_argument("--k", type=int, default=20)
a = ap.parse_args()

P = {}
for p in (l.split() for l in open(a.images_txt) if l.strip() and not l.startswith("#")):
    if len(p) < 10: continue
    qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
    Rcw = R.from_quat([qx, qy, qz, qw]).as_matrix()
    P[p[9]] = (-Rcw.T @ [tx, ty, tz], Rcw[2])  # camera center, viewing direction
A = [n for n in P if n.startswith(a.a)]; B = [n for n in P if n.startswith(a.b)]
CA = np.array([P[n][0] for n in A]); DA = np.array([P[n][1] for n in A])
pairs = set()
for n in B:
    c, d = P[n]; dist = np.linalg.norm(CA - c, axis=1); ok = (dist < a.dist) & (DA @ d > 0.5)
    for i in np.argsort(np.where(ok, dist, np.inf))[:a.k]:
        if ok[i]: pairs.add((A[i], n))
open(a.out, "w").write("".join(f"{x} {y}\n" for x, y in sorted(pairs)))
print("spatial pairs", len(pairs), "| covered", len({y for _, y in pairs}), "/", len(B))
