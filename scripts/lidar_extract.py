"""Extract Livox CustomMsg scans from an mcap bag into an npz (raw CDR parsed with numpy).

Output: xyz (N,3 float32, livox_frame), t (N float64 absolute seconds), scan (N int32 message index).
"""
import argparse
import numpy as np
from mcap.reader import make_reader

PT = np.dtype({"names": ["off", "x", "y", "z", "refl", "tag", "line"],
               "formats": ["<u4", "<f4", "<f4", "<f4", "u1", "u1", "u1"],
               "offsets": [0, 4, 8, 12, 16, 17, 18], "itemsize": 20})

def parse(buf):
    b = memoryview(buf)[4:]  # skip CDR encapsulation header; alignment is relative to here
    o = 8  # header.stamp sec/nanosec
    n = int.from_bytes(b[o:o + 4], "little"); o += 4 + n  # frame_id
    o = (o + 7) & ~7; timebase = int.from_bytes(b[o:o + 8], "little"); o += 8
    point_num = int.from_bytes(b[o:o + 4], "little"); o += 4 + 4  # point_num, lidar_id, rsvd[3]
    seq = int.from_bytes(b[o:o + 4], "little"); o += 4
    assert seq == point_num, (seq, point_num)
    return timebase, np.frombuffer(b[o:o + seq * 20], dtype=PT, count=seq)

ap = argparse.ArgumentParser()
ap.add_argument("bag"); ap.add_argument("out")
ap.add_argument("--every", type=int, default=1, help="keep every k-th scan")
ap.add_argument("--min_range", type=float, default=0.5, help="drop robot self-hits")
ap.add_argument("--max_range", type=float, default=30.0)
a = ap.parse_args()

xyz, ts, sid = [], [], []
with open(a.bag, "rb") as f:
    for k, (_, ch, msg) in enumerate(make_reader(f).iter_messages(topics=["/livox/lidar"])):
        if k % a.every: continue
        tb, p = parse(msg.data)
        P = np.stack([p["x"], p["y"], p["z"]], 1)
        r = np.linalg.norm(P, axis=1); keep = (r > a.min_range) & (r < a.max_range)
        xyz.append(P[keep]); ts.append(tb * 1e-9 + p["off"][keep] * 1e-9); sid.append(np.full(keep.sum(), k, np.int32))
xyz, ts, sid = np.concatenate(xyz), np.concatenate(ts), np.concatenate(sid)
np.savez(a.out, xyz=xyz, t=ts, scan=sid)
print(f"scans={len(np.unique(sid))} points={len(xyz)} time {ts.min():.3f}..{ts.max():.3f}")
