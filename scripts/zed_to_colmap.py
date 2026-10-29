"""Build a COLMAP text model (cameras/images, empty points) from ZED VIO poses.

T_map_optical(t) = T_map_camlink(t) * T_camlink_optical (from /tf_static).
Image ids are taken from the COLMAP database so point_triangulator can reuse its matches.
With --prefix only db images named "<prefix><frame>.jpg" are written; --align left-multiplies
a 4x4 world transform (.npy) onto every pose; --append adds to an existing model dir.
"""
import argparse, os, sqlite3
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ap = argparse.ArgumentParser()
ap.add_argument("bag"); ap.add_argument("scene")
ap.add_argument("--db"); ap.add_argument("--out")
ap.add_argument("--prefix", default="")
ap.add_argument("--align")
ap.add_argument("--append", action="store_true")
ap.add_argument("--optical", default="zed_left_camera_frame_optical")
ap.add_argument("--base", default="zed_camera_link")
a = ap.parse_args()
db_path = a.db or os.path.join(a.scene, "database.db")
out = a.out or os.path.join(a.scene, "zed_model")

def mat(tr, q):
    T = np.eye(4); T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [tr.x, tr.y, tr.z]; return T

static, ts, pos, quat = {}, [], [], []
with open(a.bag, "rb") as f:
    r = make_reader(f, decoder_factories=[DecoderFactory()])
    for _, ch, _, m in r.iter_decoded_messages(topics=["/tf_static", "/zed/zed_node/pose"]):
        if ch.topic == "/tf_static":
            for t in m.transforms:
                static[t.child_frame_id] = (t.header.frame_id, mat(t.transform.translation, t.transform.rotation))
        else:
            p, q = m.pose.position, m.pose.orientation
            ts.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            pos.append([p.x, p.y, p.z]); quat.append([q.x, q.y, q.z, q.w])

# walk static tree from optical frame up to the pose's base frame
T_base_opt, fr = np.eye(4), a.optical
while fr != a.base:
    parent, T = static[fr]; T_base_opt = T @ T_base_opt; fr = parent

ts, pos = np.array(ts), np.array(pos)
ts, order = np.unique(ts, return_index=True); pos = pos[order]  # sorted, duplicate stamps dropped
slerp = Slerp(ts, R.from_quat(np.array(quat)[order]))
T_align = np.load(a.align) if a.align else np.eye(4)

stamps = dict(l.split() for l in open(os.path.join(a.scene, "timestamps.txt")))
w, h, fx, fy, cx, cy = open(os.path.join(a.scene, "intrinsics.txt")).read().split()
db = sqlite3.connect(db_path)
ids = {n[len(a.prefix):]: (i, c) for i, n, c in db.execute("SELECT image_id, name, camera_id FROM images")
       if n.startswith(a.prefix) and n[len(a.prefix):] in stamps}

os.makedirs(out, exist_ok=True)
mode = "a" if a.append else "w"
with open(os.path.join(out, "cameras.txt"), mode) as f:
    for cid in sorted({c for _, c in ids.values()}):
        f.write(f"{cid} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
open(os.path.join(out, "points3D.txt"), mode).close()
with open(os.path.join(out, "images.txt"), mode) as f:
    for name, (iid, cid) in sorted(ids.items(), key=lambda x: x[1][0]):
        t = float(stamps[name]); t = min(max(t, ts[0]), ts[-1])
        T_map_base = np.eye(4); T_map_base[:3, :3] = slerp(t).as_matrix()
        T_map_base[:3, 3] = [np.interp(t, ts, pos[:, k]) for k in range(3)]
        T_cw = np.linalg.inv(T_align @ T_map_base @ T_base_opt)
        qx, qy, qz, qw = R.from_matrix(T_cw[:3, :3]).as_quat()
        tx, ty, tz = T_cw[:3, 3]
        f.write(f"{iid} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cid} {a.prefix}{name}\n\n")
print("wrote", len(ids), "poses; T_base_optical=\n", np.round(T_base_opt, 4))
