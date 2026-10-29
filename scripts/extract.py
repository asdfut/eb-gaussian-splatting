"""Extract sharpest ZED frame per time window from an mcap bag."""
import argparse, os
import cv2, numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ap = argparse.ArgumentParser()
ap.add_argument("bag"); ap.add_argument("out")
ap.add_argument("--fps", type=float, default=2.0)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
win = 1.0 / a.fps
best, wstart, k, intr = None, None, 0, None
stamps = open(os.path.join(a.out, "timestamps.txt"), "w")
def flush():
    global k
    if best is not None:
        cv2.imwrite(os.path.join(a.out, "images", f"{k:05d}.jpg"), best[1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        stamps.write(f"{k:05d}.jpg {best[2]:.9f}\n"); k += 1
with open(a.bag, "rb") as f:
    r = make_reader(f, decoder_factories=[DecoderFactory()])
    for _, ch, _, m in r.iter_decoded_messages(topics=["/zed/zed_node/rgb/color/rect/camera_info", "/zed/zed_node/rgb/color/rect/image/compressed"]):
        if ch.topic.endswith("camera_info"):
            intr = intr or (m.width, m.height, *m.k); continue
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if wstart is None or t - wstart >= win:
            flush(); best, wstart = None, t
        img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
        g = cv2.cvtColor(cv2.resize(img, (640, 360)), cv2.COLOR_BGR2GRAY)
        s = cv2.Laplacian(g, cv2.CV_64F).var()
        if best is None or s > best[0]: best = (s, img, t)
    flush()
stamps.close()
w, h, fx, _, cx, _, fy, cy = intr[:8]
with open(os.path.join(a.out, "intrinsics.txt"), "w") as f:
    f.write(f"{w} {h} {fx} {fy} {cx} {cy}\n")
print("saved", k, "images; intrinsics", w, h, fx, fy, cx, cy)
