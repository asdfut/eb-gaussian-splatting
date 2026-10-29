"""Estimate T_camlink_livox (Livox extrinsic) by registering LiDAR submaps to the COLMAP sparse cloud.

1) coarse: yaw grid search (Livox is mounted upright), ICP against the SfM points
2) fine: per-submap point-to-plane ICP against a LiDAR map built with the current estimate,
   averaged over many submaps spread through both sessions; iterated.
"""
import argparse, os, sys
import numpy as np, open3d as o3d
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_poses import world_camlink

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap.add_argument("--out", default="data/lidar/T_camlink_livox.npy")
ap.add_argument("--sessions", nargs="+", default=["h1", "h3"], help="mounts can differ per session")
a = ap.parse_args(); os.chdir(a.root)
EB = os.path.abspath("../eb")
BAGS = {"h1": f"{EB}/EB_try_h1_0-002.mcap", "h3": f"{EB}/EB_try-20261029T164219Z-1-001/EB_try/EB_try_h3/EB_try_h3_0.mcap"}
J = "data/joint4"
T_h1_h3 = np.load("data/joint/T_h1_h3.npy")

def pcd(P):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, np.float64)))

traj, scans = {}, {}
for b in a.sessions:
    traj[b], info = world_camlink(BAGS[b], f"data/{b}_joint4", f"{J}/final_txt/images.txt", f"{b}/",
                                  np.eye(4) if b == "h1" else T_h1_h3)
    print(b, "pose corrections:", info)
    d = np.load(f"data/lidar/{b}.npz"); scans[b] = (d["xyz"], d["t"], d["scan"])

def submap(b, s0, X, n=10, voxel=0.05):
    """n consecutive scans, deskewed per point, in world frame."""
    xyz, t, sid = scans[b]; m = (sid >= s0) & (sid < s0 + n)
    P, tt = xyz[m].astype(np.float64), t[m]
    # one pose per 10 ms bucket is plenty for a slow robot
    q = np.round(tt * 100) / 100; uq, inv = np.unique(q, return_inverse=True)
    W = traj[b](uq) @ X
    Pw = np.einsum("nij,nj->ni", W[inv, :3, :3], P) + W[inv, :3, 3]
    return pcd(Pw).voxel_down_sample(voxel)

def Xmat(yaw, t=(0, 0, 0)):
    X = np.eye(4); X[:3, :3] = R.from_euler("z", yaw, degrees=True).as_matrix(); X[:3, 3] = t; return X

sfm = pcd(np.loadtxt(f"{J}/final_txt/points3D.txt", usecols=(1, 2, 3), comments="#"))
sfm, _ = sfm.remove_statistical_outlier(20, 2.0)
probe = [p for p in [("h1", s) for s in (300, 900, 1500, 2100)] + [("h3", s) for s in (300, 1100, 1800)] if p[0] in a.sessions]

# 1) coarse yaw
scores = []
for yaw in range(0, 360, 15):
    f = []
    for b, s in probe:
        res = o3d.pipelines.registration.registration_icp(submap(b, s, Xmat(yaw)), sfm, 0.3, np.eye(4))
        f.append(res.fitness)
    scores.append((np.mean(f), yaw))
scores.sort(reverse=True); print("coarse yaw (fitness, deg):", [(round(s, 3), y) for s, y in scores[:4]])
X = Xmat(scores[0][1])

# 2) fine: each submap registered (point-to-plane) to a LiDAR map built from the *other* probes
probe = [p for p in [("h1", s) for s in range(100, 2600, 100)] + [("h3", s) for s in range(100, 2150, 100)] if p[0] in a.sessions]
for it in range(4):
    subs = {p: submap(*p, X) for p in probe}
    Xs = []
    for p in probe:
        ref = o3d.geometry.PointCloud()
        for q in probe:
            if q != p and q[0] == p[0] and abs(q[1] - p[1]) <= 600: ref += subs[q]
        if len(ref.points) < 1000: continue
        ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        res = o3d.pipelines.registration.registration_icp(
            subs[p], ref, 0.15 if it else 0.3, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        if res.fitness < 0.3: continue
        # world correction D for this submap: D @ W @ X  ->  X' = inv(W_mid) @ D @ W_mid @ X
        W_mid = traj[p[0]](scans[p[0]][1][scans[p[0]][2] == p[1]][:1])[0]
        Xs.append(np.linalg.inv(W_mid) @ res.transformation @ W_mid @ X)
    Xs = np.array(Xs)
    t_med = np.median(Xs[:, :3, 3], axis=0)
    r_med = R.from_matrix(Xs[:, :3, :3]).mean().as_matrix()
    dX = np.linalg.norm(t_med - X[:3, 3]); X = np.eye(4); X[:3, :3] = r_med; X[:3, 3] = t_med
    spread = np.median(np.linalg.norm(Xs[:, :3, 3] - t_med, axis=1))
    print(f"iter {it}: {len(Xs)} submaps, step {dX:.3f} m, spread {spread:.3f} m, "
          f"t={np.round(t_med, 3)}, rpy={np.round(R.from_matrix(r_med).as_euler('xyz', degrees=True), 2)}")
np.save(a.out, X); print("saved", a.out, "\n", np.round(X, 4))
