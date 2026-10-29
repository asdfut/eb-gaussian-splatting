"""Fuse Livox scans of both sessions into a gravity-levelled metric point cloud and reconstruct a
collision mesh that lines up with the Gaussian splatting scene.

Frames
  BA world : joint COLMAP bundle-adjusted frame (metric; what the 3DGS scene was trained in)
  sim      : BA world levelled so the floor is z = 0 (the camera paths are planar but the BA world is
             tilted ~3 deg because the ZED map frame was not gravity aligned). All outputs are in `sim`.

Steps
  - per-point deskew with the continuous trajectory; one Livox extrinsic per session
    (sessions use different sensor heights: h3 is ~1.2 m above h1)
  - robot-body removal (sensor-frame voxels persistent over a session)
  - mirror returns under the reflective floor removed with a global floor height
  - per-session free-space carving near the path (operator walking next to the robot)
  - h3 -> h1 ICP, then LiDAR -> SfM ICP so the map matches the camera/3DGS geometry
  - Poisson mesh trimmed to observed areas, decimated; flat floor grid added
"""
import argparse, os, sys, json
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_poses import world_camlink, load_colmap_c2w

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap.add_argument("--out", default="results/collision")
ap.add_argument("--voxel", type=float, default=0.02)
ap.add_argument("--poisson_depth", type=int, default=11)
ap.add_argument("--target_tris", type=int, default=300000)
ap.add_argument("--floor_cell", type=float, default=0.5)
a = ap.parse_args(); os.chdir(a.root); os.makedirs(a.out, exist_ok=True)
EB = os.path.abspath("../eb")
BAGS = {"h1": f"{EB}/EB_try_h1_0-002.mcap", "h3": f"{EB}/EB_try-20261029T164219Z-1-001/EB_try/EB_try_h3/EB_try_h3_0.mcap"}
IMAGES = "data/joint4/final_txt/images.txt"
T_h1_h3 = np.load("data/joint/T_h1_h3.npy")
XS = {b: np.load(f"data/lidar/T_camlink_livox_{b}.npy") for b in BAGS}  # calib_lidar.py --sessions b
reg = o3d.pipelines.registration
stats = {}

def pcd(P):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, np.float64)))

def rot_deg(T):
    return round(float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))), 3)

def icp(src, dst, T=np.eye(4), steps=(0.5, 0.25, 0.10)):
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    for d in steps:
        T = reg.registration_icp(src, dst, d, T, reg.TransformationEstimationPointToPlane()).transformation
    return T

# ---- levelling: robot drives on a level floor, so the h1 camera path plane is horizontal ----------
cams = load_colmap_c2w(IMAGES, "h1/"); C = np.array([T[:3, 3] for T in cams.values()])
_, _, Vt = np.linalg.svd(C - C.mean(0)); n = Vt[2] * np.sign(Vt[2][2])
L = np.eye(4); L[:3, :3] = R.align_vectors([[0, 0, 1]], [n])[0].as_matrix()   # BA world -> levelled
stats["level_tilt_removed_deg"] = round(float(np.degrees(np.arccos(n[2]))), 3)

# ---- robot body: Livox-frame voxels occupied in > half of a session's scans ----------------------
def vkey(P, s=0.05):
    v = np.floor(P / s).astype(np.int64) + 1000
    return (v[:, 0] * 2000 + v[:, 1]) * 2000 + v[:, 2]

def body_voxels(xyz, sid, frac=0.5):
    S = np.unique(sid)[::10]; m = np.isin(sid, S) & (np.linalg.norm(xyz, axis=1) < 2.0)
    uk, inv = np.unique(vkey(xyz[m]), return_inverse=True)
    pairs = np.unique(np.stack([inv, sid[m]]), axis=1)
    keys = uk[np.bincount(pairs[0], minlength=len(uk)) / len(S) > frac]
    return np.unique([kk + di * 4000000 + dj * 2000 + dk for kk in keys
                      for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)])

# ---- fuse each session (kept separate: carving must only use a session's own rays) ---------------
sess = {}
for b in BAGS:
    traj, _ = world_camlink(BAGS[b], f"data/{b}_joint4", IMAGES, f"{b}/", np.eye(4) if b == "h1" else T_h1_h3)
    d = np.load(f"data/lidar/{b}.npz"); xyz, t, sid = d["xyz"], d["t"], d["scan"]
    body = body_voxels(xyz, sid); not_body = ~np.isin(vkey(xyz), body)
    stats[f"{b}_body_points_removed"] = int((~not_body).sum())
    cl, spath, ro, re = o3d.geometry.PointCloud(), [], [], []
    for s0 in range(0, sid.max() + 1, 200):
        m = (sid >= s0) & (sid < s0 + 200) & not_body
        q = np.round(t[m] * 100) / 100; uq, inv = np.unique(q, return_inverse=True)
        W = L @ traj(uq) @ XS[b]
        Pw = np.einsum("nij,nj->ni", W[inv, :3, :3], xyz[m].astype(np.float64)) + W[inv, :3, 3]
        cl += pcd(Pw).voxel_down_sample(a.voxel); spath.append(W[:, :3, 3])
        r = np.arange(0, m.sum(), 20); ro.append(W[inv[r], :3, 3]); re.append(Pw[r])  # rays for carving
    sess[b] = dict(cloud=cl.voxel_down_sample(a.voxel), path=np.concatenate(spath)[::5],
                   O=np.concatenate(ro), E=np.concatenate(re))
    stats[f"{b}_sensor_height_z_median"] = round(float(np.median(sess[b]["path"][:, 2])), 3)
    print(b, "fused:", len(sess[b]["cloud"].points), "sensor z median", stats[f"{b}_sensor_height_z_median"], flush=True)
stats["points_fused"] = sum(len(v["cloud"].points) for v in sess.values())

# ---- global floor height: densest z level just below the low (h1) sensor, 1-4 m from its path ----
P1, p1 = np.asarray(sess["h1"]["cloud"].points), sess["h1"]["path"]
dist, _ = cKDTree(p1[:, :2]).query(P1[:, :2]); zs = np.median(p1[:, 2])
hist, edges = np.histogram(P1[(dist > 1.0) & (dist < 4.0), 2], bins=np.arange(zs - 0.6, zs + 0.01, 0.01))
floor_z = edges[np.argmax(hist)] + 0.005
stats["floor_z"] = round(float(floor_z), 3); stats["h1_sensor_above_floor_m"] = round(float(zs - floor_z), 3)

# ---- per session: mirror removal + free-space carving -------------------------------------------
CV = 0.10
def ckey(P):
    v = np.floor(P / CV).astype(np.int64) + 1000          # +-100 m at 10 cm
    return (v[:, 0] * 2000 + v[:, 1]) * 2000 + v[:, 2]

def lookup(keys, counts, q):
    out = np.zeros(len(q)); pos = np.searchsorted(keys, q)
    ok = (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == q); out[ok] = counts[pos[ok]]
    return out

carved = []
for b, v in sess.items():
    cl = v["cloud"]; Pb = np.asarray(cl.points)
    keep = Pb[:, 2] > floor_z - 0.05          # returns mirrored by the shiny floor lie below it
    stats[f"{b}_mirror_points_removed"] = int((~keep).sum())
    cl = cl.select_by_index(np.where(keep)[0]); Pb = Pb[keep]
    db, _ = cKDTree(v["path"][:, :2]).query(Pb[:, :2])
    # a voxel many rays pass through but few end in held a moving object (the operator next to the
    # robot). Rays ending below floor+10 cm are skipped (mirror returns pass *through* the floor);
    # only space >= 15 cm above the floor and within 1.5 m of the path is carved.
    O, E = v["O"], v["E"]; above = E[:, 2] > floor_z + 0.10; O, E = O[above], E[above]
    hk, hc = np.unique(ckey(E), return_counts=True)
    free_keys = []
    for i in range(0, len(O), 50000):
        o, e = O[i:i + 50000], E[i:i + 50000]
        dv = e - o; Ln = np.linalg.norm(dv, axis=1); u = dv / Ln[:, None]
        sm = np.arange(0.3, 4.0, CV / 2)
        S = o[:, None, :] + u[:, None, :] * sm[None, :, None]
        ok = (sm[None, :] < (Ln[:, None] - 0.25)) & (S[..., 2] > floor_z + 0.15)
        k = ckey(S[ok]); ray_id = np.repeat(np.arange(len(o)), ok.sum(1))
        free_keys.append(np.unique(k * 65536 + ray_id) // 65536)  # count each ray once per voxel
    fk, fc = np.unique(np.concatenate(free_keys), return_counts=True)
    kb = ckey(Pb); f_at, h_at = lookup(fk, fc, kb), lookup(hk, hc, kb)
    is_dyn = (db < 1.5) & (f_at >= 5) & (f_at > 3 * h_at)
    carved.append(np.c_[Pb[is_dyn], f_at[is_dyn], h_at[is_dyn], db[is_dyn]].astype(np.float32))
    stats[f"{b}_dynamic_points_removed"] = int(is_dyn.sum())
    v["final"] = cl.select_by_index(np.where(~is_dyn)[0])
    print(b, "mirror removed", stats[f"{b}_mirror_points_removed"], "carved", int(is_dyn.sum()), flush=True)
np.savez(f"{a.out}/carved_points.npz", pts=np.concatenate(carved))  # x, y, z, free, hit, dist-to-path

# ---- session alignment, then LiDAR -> SfM (camera / 3DGS geometry) ------------------------------
src, dst = sess["h3"]["final"].voxel_down_sample(0.05), sess["h1"]["final"].voxel_down_sample(0.05)
ev0 = reg.evaluate_registration(src, dst, 0.10); T13 = icp(src, dst); ev1 = reg.evaluate_registration(src, dst, 0.10, T13)
stats["h3_to_h1_icp"] = {"translation_m": np.round(T13[:3, 3], 3).tolist(), "rotation_deg": rot_deg(T13),
                         "overlap@10cm": [round(ev0.fitness, 3), round(ev1.fitness, 3)],
                         "rmse_m": [round(ev0.inlier_rmse, 4), round(ev1.inlier_rmse, 4)]}
cloud = (sess["h1"]["final"] + sess["h3"]["final"].transform(T13)).voxel_down_sample(a.voxel)
path = np.concatenate([sess["h1"]["path"], sess["h3"]["path"] @ T13[:3, :3].T + T13[:3, 3]])

sfm = pcd((L[:3, :3] @ np.loadtxt("data/joint4/final_txt/points3D.txt", usecols=(1, 2, 3), comments="#").T).T)
sfm, _ = sfm.remove_statistical_outlier(20, 2.0)
lid = cloud.voxel_down_sample(0.05)
e0 = [reg.evaluate_registration(sfm, lid, d) for d in (0.05, 0.10)]
Ts = icp(sfm, lid)                              # SfM -> LiDAR
e1 = [reg.evaluate_registration(sfm, lid, d, Ts) for d in (0.05, 0.10)]
G = np.linalg.inv(Ts)                           # LiDAR -> SfM/3DGS
stats["lidar_to_sfm"] = {"translation_m": np.round(G[:3, 3], 3).tolist(), "rotation_deg": rot_deg(G),
                         "sfm_within_5cm": [round(e0[0].fitness, 3), round(e1[0].fitness, 3)],
                         "sfm_within_10cm": [round(e0[1].fitness, 3), round(e1[1].fitness, 3)]}
cloud.transform(G); path = path @ G[:3, :3].T + G[:3, 3]
floor_z = float(floor_z + G[2, 3])              # G is ~a translation (rotation < 1 deg)
# put the floor at z = 0
Z = np.eye(4); Z[2, 3] = -floor_z; cloud.transform(Z); path[:, 2] -= floor_z
print("h3->h1", stats["h3_to_h1_icp"], "\nlidar->sfm", stats["lidar_to_sfm"], flush=True)

cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
stats["points_final"] = len(cloud.points)
o3d.io.write_point_cloud(f"{a.out}/lidar_map.ply", cloud)

# BA world -> sim and sim -> gsplat normalized frame (for overlaying the 3DGS .ply)
T_sim_from_ba = Z @ G @ L
T_gs_from_ba = np.loadtxt("data/lidar/T_gsplat_from_world.txt")
np.savetxt(f"{a.out}/T_sim_from_ba.txt", T_sim_from_ba, fmt="%.8f", header="x_sim = T @ x_ba (BA world = COLMAP sparse model / 3DGS training frame before normalization)")
np.savetxt(f"{a.out}/T_gsplat_from_sim.txt", T_gs_from_ba @ np.linalg.inv(T_sim_from_ba), fmt="%.8f",
           header="x_gsplat = T @ x_sim (gsplat normalized frame of the exported .ply / checkpoints)")

# ---- mesh ----------------------------------------------------------------------------------------
cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
P, N = np.asarray(cloud.points), np.asarray(cloud.normals)
_, j = cKDTree(path).query(P); flip = np.einsum("ij,ij->i", N, path[j] - P) < 0
N[flip] *= -1; cloud.normals = o3d.utility.Vector3dVector(N)
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=a.poisson_depth, n_threads=-1)
dens = np.asarray(dens); mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.03))
dv, _ = cKDTree(P).query(np.asarray(mesh.vertices)); mesh.remove_vertices_by_mask(dv > 3 * a.voxel)  # Poisson inventions
mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
tri_c, n_c, _ = mesh.cluster_connected_triangles(); tri_c, n_c = np.asarray(tri_c), np.asarray(n_c)
mesh.remove_triangles_by_mask(n_c[tri_c] < 200); mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()
stats["mesh_triangles"] = len(mesh.triangles)
o3d.io.write_triangle_mesh(f"{a.out}/collision_mesh_full.ply", mesh)
simp = mesh.simplify_quadric_decimation(a.target_tris); simp.remove_unreferenced_vertices(); simp.compute_vertex_normals()
o3d.io.write_triangle_mesh(f"{a.out}/collision_mesh_nofloor.obj", simp, write_vertex_normals=True)

# flat floor grid at z = 0 over the mesh footprint: the 9 cm high Mid-360 sees the shiny floor only at
# grazing angles, so floor returns are too sparse for Poisson to close it
V = np.asarray(simp.vertices); lo, hi = V[:, :2].min(0) - 0.5, V[:, :2].max(0) + 0.5
xs = np.arange(lo[0], hi[0] + a.floor_cell, a.floor_cell); ys = np.arange(lo[1], hi[1] + a.floor_cell, a.floor_cell)
gx, gy = np.meshgrid(xs, ys, indexing="ij"); idx = np.arange(gx.size).reshape(gx.shape)
q = np.c_[idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()]
floor = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]),
                                  o3d.utility.Vector3iVector(np.r_[q[:, [0, 1, 2]], q[:, [0, 2, 3]]]))
o3d.io.write_triangle_mesh(f"{a.out}/floor.obj", floor)
merged = simp + floor; merged.compute_vertex_normals()
o3d.io.write_triangle_mesh(f"{a.out}/collision_mesh.obj", merged, write_vertex_normals=True)
stats["collision_triangles"] = len(merged.triangles); stats["floor_triangles"] = len(floor.triangles)
bb = cloud.get_axis_aligned_bounding_box(); stats["bbox_min"] = bb.min_bound.round(2).tolist(); stats["bbox_max"] = bb.max_bound.round(2).tolist()
json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=2); print(json.dumps(stats))
