"""Continuous-time zed_camera_link poses in the joint COLMAP (final BA) world frame.

T_world_camlink(t) = C(t) @ T_align @ T_map_camlink(t)
  T_map_camlink : ZED VIO pose from the bag (30 Hz, slerp)
  T_align       : h3 -> h1 rigid transform (identity for h1)
  C(t)          : per-image BA correction (final_c2w @ inv(zed_c2w)), interpolated over time,
                  so LiDAR maps line up with the Gaussian splatting scene.
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

def _mat(tr, q):
    T = np.eye(4); T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [tr.x, tr.y, tr.z]; return T

class Traj:
    """Rigid-body trajectory sampled at times ts (Nx4x4), queried with slerp + linear interpolation."""
    def __init__(self, ts, T):
        ts, idx = np.unique(np.asarray(ts), return_index=True); T = np.asarray(T)[idx]
        self.ts, self.p = ts, T[:, :3, 3]; self.slerp = Slerp(ts, R.from_matrix(T[:, :3, :3]))
    def __call__(self, t):
        t = np.clip(np.atleast_1d(t), self.ts[0], self.ts[-1])
        out = np.tile(np.eye(4), (len(t), 1, 1)); out[:, :3, :3] = self.slerp(t).as_matrix()
        for k in range(3): out[:, k, 3] = np.interp(t, self.ts, self.p[:, k])
        return out

def load_zed(bag, optical="zed_left_camera_frame_optical", base="zed_camera_link"):
    static, ts, T = {}, [], []
    with open(bag, "rb") as f:
        for _, ch, _, m in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=["/tf_static", "/zed/zed_node/pose"]):
            if ch.topic == "/tf_static":
                for t in m.transforms:
                    static[t.child_frame_id] = (t.header.frame_id, _mat(t.transform.translation, t.transform.rotation))
            else:
                ts.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
                T.append(_mat(m.pose.position, m.pose.orientation))
    T_base_opt, fr = np.eye(4), optical
    while fr != base:
        parent, X = static[fr]; T_base_opt = X @ T_base_opt; fr = parent
    return Traj(ts, T), T_base_opt

def load_colmap_c2w(images_txt, prefix):
    out, L = {}, [l for l in open(images_txt) if not l.startswith("#")]
    for l in L[::2]:
        s = l.split()
        if len(s) < 10 or not s[9].startswith(prefix): continue
        qw, qx, qy, qz, tx, ty, tz = map(float, s[1:8])
        Rcw = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4); T[:3, :3] = Rcw.T; T[:3, 3] = -Rcw.T @ [tx, ty, tz]
        out[s[9][len(prefix):]] = T
    return out

def world_camlink(bag, scene_dir, images_txt, prefix, T_align=np.eye(4), max_corr=0.3):
    """Returns a Traj of T_world_camlink in the final BA frame, plus correction stats."""
    zed, T_base_opt = load_zed(bag)
    stamps = dict(l.split() for l in open(os.path.join(scene_dir, "timestamps.txt")))
    final = load_colmap_c2w(images_txt, prefix)
    ts, C = [], []
    for name, T_f in final.items():
        t = float(stamps[name]); T_z = T_align @ zed(t)[0] @ T_base_opt
        ts.append(t); C.append(T_f @ np.linalg.inv(T_z))
    ts, C = np.array(ts), np.array(C); o = np.argsort(ts); ts, C = ts[o], C[o]
    # drop images whose correction jumps away from its neighbours (bad BA poses)
    med = np.array([np.median(C[max(0, i - 10):i + 11, :3, 3], axis=0) for i in range(len(C))])
    good = np.linalg.norm(C[:, :3, 3] - med, axis=1) < max_corr
    corr = Traj(ts[good], C[good])
    zts = zed.ts
    W = corr(zts) @ (T_align @ zed(zts))
    return Traj(zts, W), {"images": len(ts), "dropped": int((~good).sum()),
                          "corr_median_m": float(np.median(np.linalg.norm(C[good, :3, 3] - C[good, :3, 3].mean(0), axis=1)))}
